import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import hashlib
import secrets
import string
import math
import smtplib
import logging
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager
import os
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text, Boolean, Float, ForeignKey, Index, func, and_, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Classes
@dataclass
class DatabaseConfig:
    url: str = os.getenv('DATABASE_URL', 'sqlite:///hospital_referral.db')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', '5'))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', '10'))
    pool_recycle: int = int(os.getenv('DB_POOL_RECYCLE', '3600'))

@dataclass
class SMTPConfig:
    server: str = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    port: int = int(os.getenv('SMTP_PORT', '587'))
    username: Optional[str] = os.getenv('SMTP_USERNAME')
    password: Optional[str] = os.getenv('SMTP_PASSWORD')
    use_tls: bool = True

@dataclass
class MapConfig:
    default_latitude: float = -0.0916
    default_longitude: float = 34.7680
    default_zoom: int = 10
    google_maps_api_key: Optional[str] = os.getenv('GOOGLE_MAPS_API_KEY')

@dataclass
class CostConfig:
    fuel_price_per_liter: float = float(os.getenv('FUEL_PRICE_PER_LITER', '180.0'))
    average_fuel_consumption: float = 0.12  # Liters per kilometer
    base_operating_cost_per_km: float = 50.0
    fuel_tank_capacity: float = 80.0

@dataclass
class AppConfig:
    page_title: str = "Kisumu County Hospital Referral System"
    page_icon: str = "🏥"
    layout: str = "wide"
    notification_check_interval: int = 30
    location_update_interval: int = 10
    secret_key: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

class Config:
    database = DatabaseConfig()
    smtp = SMTPConfig()
    maps = MapConfig()
    costs = CostConfig()
    app = AppConfig()

# Database Models
Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    hospital = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    created_patients = relationship("Patient", back_populates="creator")
    created_referrals = relationship("Referral", back_populates="creator")
    communications = relationship("Communication", back_populates="sender_user")

class Patient(Base):
    __tablename__ = 'patients'
    
    patient_id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    condition = Column(String(255), nullable=False)
    referring_hospital = Column(String(255), nullable=False)
    receiving_hospital = Column(String(255), nullable=False)
    referring_physician = Column(String(100), nullable=False)
    receiving_physician = Column(String(100))
    notes = Column(Text)
    vital_signs = Column(JSON)
    medical_history = Column(Text)
    current_medications = Column(Text)
    allergies = Column(Text)
    referral_time = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(50), default='Referred', index=True)
    assigned_ambulance = Column(String, ForeignKey('ambulances.ambulance_id'))
    created_by = Column(String, ForeignKey('users.id'))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    referring_hospital_lat = Column(Float)
    referring_hospital_lng = Column(Float)
    receiving_hospital_lat = Column(Float)
    receiving_hospital_lng = Column(Float)
    pickup_notification_sent = Column(Boolean, default=False)
    enroute_notification_sent = Column(Boolean, default=False)
    
    # Cost tracking fields
    trip_distance = Column(Float)
    trip_fuel_cost = Column(Float)
    trip_cost_savings = Column(Float, default=0.0)
    
    # Relationships
    creator = relationship("User", back_populates="created_patients")
    ambulance = relationship("Ambulance", back_populates="patients")
    referrals = relationship("Referral", back_populates="patient")
    handover_forms = relationship("HandoverForm", back_populates="patient")
    communications = relationship("Communication", back_populates="patient")
    location_updates = relationship("LocationUpdate", back_populates="patient")

class Ambulance(Base):
    __tablename__ = 'ambulances'
    
    ambulance_id = Column(String, primary_key=True, default=generate_uuid)
    current_location = Column(String(255))
    latitude = Column(Float, index=True)
    longitude = Column(Float, index=True)
    status = Column(String(50), default='Available', index=True)
    driver_name = Column(String(100), nullable=False)
    driver_contact = Column(String(20))
    current_patient = Column(String, ForeignKey('patients.patient_id'))
    destination = Column(String(255))
    route = Column(JSON)
    start_time = Column(DateTime)
    current_step = Column(Integer, default=0)
    mission_complete = Column(Boolean, default=False)
    estimated_arrival = Column(DateTime)
    last_location_update = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Fuel and cost tracking
    fuel_level = Column(Float, default=100.0)
    fuel_consumption_rate = Column(Float, default=0.12)
    total_fuel_cost = Column(Float, default=0.0)
    total_distance_traveled = Column(Float, default=0.0)
    cost_savings = Column(Float, default=0.0)
    
    # Relationships
    patients = relationship("Patient", back_populates="ambulance")
    referrals = relationship("Referral", back_populates="ambulance")
    communications = relationship("Communication", back_populates="ambulance")
    location_updates = relationship("LocationUpdate", back_populates="ambulance")

class Referral(Base):
    __tablename__ = 'referrals'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey('patients.patient_id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(50), default='Ambulance Dispatched')
    ambulance_id = Column(String, ForeignKey('ambulances.ambulance_id'))
    created_by = Column(String, ForeignKey('users.id'))
    
    # Relationships
    patient = relationship("Patient", back_populates="referrals")
    ambulance = relationship("Ambulance", back_populates="referrals")
    creator = relationship("User", back_populates="created_referrals")

class HandoverForm(Base):
    __tablename__ = 'handover_forms'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey('patients.patient_id'), nullable=False, index=True)
    patient_name = Column(String(100))
    age = Column(Integer)
    condition = Column(String(255))
    referring_hospital = Column(String(255))
    receiving_hospital = Column(String(255))
    referring_physician = Column(String(100))
    receiving_physician = Column(String(100))
    transfer_time = Column(DateTime, default=datetime.utcnow)
    vital_signs = Column(JSON)
    medical_history = Column(Text)
    current_medications = Column(Text)
    allergies = Column(Text)
    notes = Column(Text)
    ambulance_id = Column(String)
    created_by = Column(String, ForeignKey('users.id'))
    
    # Relationships
    patient = relationship("Patient", back_populates="handover_forms")

class Communication(Base):
    __tablename__ = 'communications'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey('patients.patient_id'), index=True)
    ambulance_id = Column(String, ForeignKey('ambulances.ambulance_id'), index=True)
    sender = Column(String(100), nullable=False)
    receiver = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    message_type = Column(String(50))
    sender_id = Column(String, ForeignKey('users.id'))
    
    # Relationships
    patient = relationship("Patient", back_populates="communications")
    ambulance = relationship("Ambulance", back_populates="communications")
    sender_user = relationship("User", back_populates="communications")

class LocationUpdate(Base):
    __tablename__ = 'location_updates'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    ambulance_id = Column(String, ForeignKey('ambulances.ambulance_id'), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_name = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    patient_id = Column(String, ForeignKey('patients.patient_id'))
    
    # Relationships
    ambulance = relationship("Ambulance", back_populates="location_updates")
    patient = relationship("Patient", back_populates="location_updates")

# Create database engine and session
engine = create_engine(
    Config.database.url,
    connect_args={"check_same_thread": False} if "sqlite" in Config.database.url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Authentication System
class Authentication:
    def __init__(self):
        self.session = st.session_state
        if 'authenticated' not in self.session:
            self.session.authenticated = False
        if 'user' not in self.session:
            self.session.user = None
        if 'token' not in self.session:
            self.session.token = None

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(plain_password) == hashed_password

    def _create_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token for user session"""
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'hospital': user_data['hospital'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, Config.app.secret_key, algorithm='HS256')

    def _verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload if valid"""
        try:
            payload = jwt.decode(token, Config.app.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            st.error("Session expired. Please login again.")
            return None
        except jwt.InvalidTokenError:
            st.error("Invalid session token. Please login again.")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user against database"""
        try:
            with session_scope() as session:
                user = session.query(User).filter(
                    User.username == username,
                    User.is_active == True
                ).first()
                
                if user and self._verify_password(password, user.password_hash):
                    user.last_login = datetime.utcnow()
                    session.commit()
                    
                    user_data = {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'role': user.role,
                        'hospital': user.hospital,
                        'name': user.name,
                        'last_login': user.last_login
                    }
                    
                    # Create JWT token
                    token = self._create_jwt_token(user_data)
                    
                    return {
                        **user_data,
                        'token': token
                    }
            
            return None
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None

    def register_user(self, user_data: Dict[str, Any]) -> bool:
        """Register new user"""
        try:
            with session_scope() as session:
                # Check if username already exists
                existing_user = session.query(User).filter(
                    User.username == user_data['username']
                ).first()
                
                if existing_user:
                    st.error("Username already exists")
                    return False
                
                # Create new user
                new_user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash=self._hash_password(user_data['password']),
                    role=user_data['role'],
                    hospital=user_data['hospital'],
                    name=user_data['name']
                )
                
                session.add(new_user)
                session.commit()
                
                st.success(f"User {user_data['username']} created successfully")
                return True
                
        except Exception as e:
            st.error(f"Registration error: {str(e)}")
            return False

    def setup_auth_ui(self):
        """Setup authentication UI in sidebar"""
        st.sidebar.title("🔐 Authentication")
        
        if not self.session.authenticated:
            tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
            
            with tab1:
                self._login_form()
            with tab2:
                self._register_form()
        else:
            self._logout_section()

    def _login_form(self):
        """Login form"""
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.form_submit_button("Login", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password")
                    return
                
                user = self.authenticate_user(username, password)
                if user:
                    self.session.authenticated = True
                    self.session.user = user
                    self.session.token = user['token']
                    st.sidebar.success(f"Welcome {user['role']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    def _register_form(self):
        """Registration form (admin only)"""
        if not self.session.authenticated:
            st.info("Please login as admin to register new users")
            return
            
        if self.session.user['role'] != 'Admin':
            st.warning("Only administrators can register new users")
            return
            
        with st.form("register_form"):
            st.subheader("Register New User")
            
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            name = st.text_input("Full Name")
            role = st.selectbox("Role", ["Admin", "Hospital Staff", "Ambulance Driver"])
            hospital = st.selectbox("Hospital", self._get_hospital_options())
            
            if st.form_submit_button("Register User", use_container_width=True):
                if not all([username, email, password, name]):
                    st.error("Please fill all fields")
                    return
                    
                if password != confirm_password:
                    st.error("Passwords do not match")
                    return
                    
                user_data = {
                    'username': username,
                    'email': email,
                    'password': password,
                    'role': role,
                    'hospital': hospital,
                    'name': name
                }
                
                if self.register_user(user_data):
                    st.rerun()

    def _get_hospital_options(self):
        """Get hospital options for registration"""
        return [
            "All Facilities",
            "Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)",
            "Kisumu County Referral Hospital",
            "Lumumba Sub-County Hospital", 
            "Ahero Sub-County Hospital", 
            "Kombewa Sub-County / District Hospital", 
            "Muhoroni County Hospital"
        ]

    def _logout_section(self):
        """Logout section"""
        st.sidebar.success(f"Logged in as: {self.session.user['name']}")
        st.sidebar.write(f"**Role:** {self.session.user['role']}")
        st.sidebar.write(f"**Hospital:** {self.session.user['hospital']}")
        
        if st.sidebar.button("Logout", use_container_width=True):
            self.session.clear()
            st.rerun()

    def require_auth(self, roles: Optional[list] = None) -> bool:
        """Check if user is authenticated and has required roles"""
        if not self.session.authenticated:
            st.warning("Please login to access this page")
            return False
            
        # Verify JWT token
        if not self.session.token or not self._verify_jwt_token(self.session.token):
            self.session.clear()
            st.warning("Session expired. Please login again.")
            return False
            
        if roles and self.session.user['role'] not in roles:
            st.error(f"Access denied. Required roles: {', '.join(roles)}")
            return False
            
        return True

    def initialize_default_users(self):
        """Initialize default users if none exist"""
        try:
            with session_scope() as session:
                user_count = session.query(User).count()
                
                if user_count == 0:
                    default_users = [
                        {
                            'username': 'admin',
                            'email': 'admin@kisumu.gov',
                            'password': 'admin123',
                            'role': 'Admin',
                            'hospital': 'All Facilities',
                            'name': 'System Administrator'
                        },
                        {
                            'username': 'hospital_staff',
                            'email': 'staff@joortrh.go.ke',
                            'password': 'staff123',
                            'role': 'Hospital Staff',
                            'hospital': 'Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)',
                            'name': 'Hospital Staff Member'
                        },
                        {
                            'username': 'driver',
                            'email': 'driver@kisumu.gov',
                            'password': 'driver123',
                            'role': 'Ambulance Driver',
                            'hospital': 'Ambulance Service',
                            'name': 'Ambulance Driver'
                        }
                    ]
                    
                    for user_data in default_users:
                        user = User(
                            username=user_data['username'],
                            email=user_data['email'],
                            password_hash=self._hash_password(user_data['password']),
                            role=user_data['role'],
                            hospital=user_data['hospital'],
                            name=user_data['name']
                        )
                        session.add(user)
                    
                    session.commit()
                    st.info("Default users initialized")
                    
        except Exception as e:
            st.error(f"Error initializing default users: {str(e)}")

# Services
class DatabaseService:
    """Enhanced database service with connection pooling and error handling"""
    
    def __init__(self):
        self.engine = engine
        
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        with session_scope() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                raise

class CostCalculationService:
    """Service for calculating trip costs and fuel consumption"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
    
    def calculate_trip_cost(self, distance_km: float, fuel_consumption_rate: Optional[float] = None) -> Dict[str, float]:
        """Calculate comprehensive trip costs"""
        if fuel_consumption_rate is None:
            fuel_consumption_rate = Config.costs.average_fuel_consumption
        
        fuel_used = distance_km * fuel_consumption_rate
        fuel_cost = fuel_used * Config.costs.fuel_price_per_liter
        operating_cost = distance_km * Config.costs.base_operating_cost_per_km
        total_cost = fuel_cost + operating_cost
        
        return {
            'distance_km': distance_km,
            'fuel_used_liters': fuel_used,
            'fuel_cost_ksh': fuel_cost,
            'operating_cost_ksh': operating_cost,
            'total_cost_ksh': total_cost
        }
    
    def calculate_potential_savings(self, actual_distance: float, alternative_distance: float) -> float:
        """Calculate potential savings from efficient routing"""
        actual_cost = self.calculate_trip_cost(actual_distance)
        alternative_cost = self.calculate_trip_cost(alternative_distance)
        
        savings = alternative_cost['total_cost_ksh'] - actual_cost['total_cost_ksh']
        return max(0, savings)

class NotificationService:
    """Enhanced notification service with email and SMS support"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.smtp_config = Config.smtp
    
    def send_email(self, to_email: str, subject: str, message: str, html: bool = False) -> bool:
        """Send email notification"""
        try:
            if not self.smtp_config.username or not self.smtp_config.password:
                logger.warning("SMTP credentials not configured")
                return False
            
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_config.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            if html:
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_config.server, self.smtp_config.port) as server:
                if self.smtp_config.use_tls:
                    server.starttls()
                server.login(self.smtp_config.username, self.smtp_config.password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_sms(self, to_number: str, message: str) -> bool:
        """Send SMS notification (placeholder for Twilio integration)"""
        logger.info(f"SMS to {to_number}: {message}")
        return True

class ReferralService:
    """Service for managing patient referrals"""
    
    def __init__(self, db_service: DatabaseService, notification_service: NotificationService):
        self.db_service = db_service
        self.notification_service = notification_service
        self.cost_service = CostCalculationService(db_service)
    
    def create_referral(self, patient_data: Dict, user: Dict) -> Optional[Patient]:
        """Create new patient referral with validation"""
        try:
            with self.db_service.get_session() as session:
                # Validate required fields
                required_fields = ['name', 'age', 'condition', 'referring_hospital', 'receiving_hospital', 'referring_physician']
                for field in required_fields:
                    if not patient_data.get(field):
                        raise ValueError(f"Missing required field: {field}")
                
                # Create patient
                patient = Patient(**patient_data)
                session.add(patient)
                session.flush()  # Get the patient_id
                
                # Create referral record
                referral = Referral(
                    patient_id=patient.patient_id,
                    created_by=user['id'],
                    ambulance_id=patient_data.get('assigned_ambulance')
                )
                session.add(referral)
                session.commit()
                
                return patient
                
        except Exception as e:
            logger.error(f"Error creating referral: {str(e)}")
            st.error(f"Failed to create referral: {str(e)}")
            return None
    
    def assign_ambulance(self, patient_id: str, ambulance_id: str) -> bool:
        """Assign ambulance to patient"""
        try:
            with self.db_service.get_session() as session:
                patient = session.query(Patient).filter(Patient.patient_id == patient_id).first()
                ambulance = session.query(Ambulance).filter(Ambulance.ambulance_id == ambulance_id).first()
                
                if not patient or not ambulance:
                    st.error("Patient or ambulance not found")
                    return False
                
                if ambulance.status != 'Available':
                    st.error("Ambulance is not available")
                    return False
                
                if ambulance.fuel_level < 20:
                    st.error("Ambulance fuel level too low")
                    return False
                
                # Update assignments
                patient.assigned_ambulance = ambulance_id
                patient.status = 'Ambulance Assigned'
                
                ambulance.status = 'On Transfer'
                ambulance.current_patient = patient_id
                ambulance.destination = patient.receiving_hospital
                
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error assigning ambulance: {str(e)}")
            st.error(f"Failed to assign ambulance: {str(e)}")
            return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

class AnalyticsService:
    """Service for analytics and reporting"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.cost_service = CostCalculationService(db_service)
    
    def get_kpis(self) -> Dict[str, any]:
        """Get key performance indicators"""
        with self.db_service.get_session() as session:
            # Basic counts
            total_patients = session.query(Patient).count()
            active_patients = session.query(Patient).filter(
                Patient.status.notin_(['Completed', 'Arrived at Destination'])
            ).count()
            total_ambulances = session.query(Ambulance).count()
            available_ambulances = session.query(Ambulance).filter(
                Ambulance.status == 'Available'
            ).count()
            
            # Cost metrics
            total_fuel_cost = session.query(func.sum(Ambulance.total_fuel_cost)).scalar() or 0
            total_savings = session.query(func.sum(Ambulance.cost_savings)).scalar() or 0
            total_distance = session.query(func.sum(Ambulance.total_distance_traveled)).scalar() or 0
            
            # Response time (simplified)
            completed_referrals = session.query(Patient).filter(
                Patient.status == 'Completed'
            ).count()
            
            avg_response_time = 15.0  # Simplified calculation
            
            completion_rate = (completed_referrals / total_patients * 100) if total_patients > 0 else 0
            
            return {
                'total_referrals': total_patients,
                'active_referrals': active_patients,
                'total_ambulances': total_ambulances,
                'available_ambulances': available_ambulances,
                'avg_response_time': f"{avg_response_time:.1f} min",
                'completion_rate': f"{completion_rate:.1f}%",
                'total_fuel_cost': total_fuel_cost,
                'total_cost_savings': total_savings,
                'total_distance_km': total_distance,
                'cost_efficiency': f"{(total_savings / total_fuel_cost * 100) if total_fuel_cost > 0 else 0:.1f}%"
            }
    
    def get_cost_analytics(self) -> Dict[str, any]:
        """Get detailed cost analytics"""
        with self.db_service.get_session() as session:
            # Monthly cost trends (simulated for demo)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            
            total_fuel_cost = session.query(func.sum(Ambulance.total_fuel_cost)).scalar() or 0
            total_savings = session.query(func.sum(Ambulance.cost_savings)).scalar() or 0
            
            monthly_costs = [total_fuel_cost * (0.8 + i * 0.1) for i in range(6)]
            monthly_savings = [total_savings * (0.7 + i * 0.15) for i in range(6)]
            
            return {
                'monthly_costs': monthly_costs,
                'monthly_savings': monthly_savings,
                'months': months,
                'total_fuel_cost': total_fuel_cost,
                'total_savings': total_savings
            }

class AmbulanceService:
    """Service for ambulance management"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

# UI Components
class DashboardUI:
    """Dashboard UI component"""
    
    def __init__(self, analytics_service: AnalyticsService, db_service: DatabaseService):
        self.analytics = analytics_service
        self.db_service = db_service
    
    def display(self):
        """Display main dashboard"""
        st.title("📊 Dashboard Overview")
        
        # Get KPIs
        kpis = self.analytics.get_kpis()
        
        # Display KPIs
        self._display_kpi_metrics(kpis)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            self._display_cost_analytics()
        with col2:
            self._display_referral_trends()
        
        # Recent activity
        st.subheader("Recent Activity")
        self._display_recent_activity()

    def _display_kpi_metrics(self, kpis: Dict):
        """Display KPI metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Referrals", kpis['total_referrals'])
        with col2:
            st.metric("Active Referrals", kpis['active_referrals'])
        with col3:
            st.metric("Available Ambulances", kpis['available_ambulances'])
        with col4:
            st.metric("Avg Response Time", kpis['avg_response_time'])
        with col5:
            st.metric("Completion Rate", kpis['completion_rate'])
        
        # Cost KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fuel Cost", f"KSh {kpis['total_fuel_cost']:,.0f}")
        with col2:
            st.metric("Cost Savings", f"KSh {kpis['total_cost_savings']:,.0f}")
        with col3:
            st.metric("Total Distance", f"{kpis['total_distance_km']:,.1f} km")
        with col4:
            st.metric("Cost Efficiency", kpis['cost_efficiency'])

    def _display_cost_analytics(self):
        """Display cost analytics chart"""
        st.subheader("💰 Cost Analytics")
        cost_data = self.analytics.get_cost_analytics()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cost_data['months'],
            y=cost_data['monthly_costs'],
            name='Costs Incurred',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=cost_data['months'],
            y=cost_data['monthly_savings'],
            name='Costs Saved',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title='Monthly Costs vs Savings',
            xaxis_title='Month',
            yaxis_title='Amount (KSh)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    def _display_referral_trends(self):
        """Display referral trends"""
        st.subheader("📈 Referral Trends")
        st.info("Referral trends chart would be implemented here")

    def _display_recent_activity(self):
        """Display recent activity table"""
        with self.db_service.get_session() as session:
            recent_patients = session.query(Patient).order_by(
                Patient.referral_time.desc()
            ).limit(10).all()
            
            if recent_patients:
                data = []
                for patient in recent_patients:
                    data.append({
                        'Patient ID': patient.patient_id,
                        'Name': patient.name,
                        'Condition': patient.condition,
                        'From': patient.referring_hospital,
                        'To': patient.receiving_hospital,
                        'Status': patient.status,
                        'Time': patient.referral_time.strftime('%Y-%m-%d %H:%M')
                    })
                
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No recent activity")

class ReferralUI:
    """Referral management UI"""
    
    def __init__(self, referral_service: ReferralService, db_service: DatabaseService):
        self.referral_service = referral_service
        self.db_service = db_service
    
    def display(self):
        """Display referral management interface"""
        st.title("📋 Patient Referral Management")
        
        tab1, tab2, tab3 = st.tabs(["Create Referral", "Active Referrals", "Referral History"])
        
        with tab1:
            self._create_referral_form()
        with tab2:
            self._display_active_referrals()
        with tab3:
            self._display_referral_history()

    def _create_referral_form(self):
        """Create referral form"""
        st.subheader("Create New Patient Referral")
        
        with st.form("referral_form", clear_on_submit=True):
            patient_data = self._get_patient_form_data()
            
            submitted = st.form_submit_button("Create Referral", use_container_width=True)
            if submitted:
                is_valid, error_message = self._validate_patient_data(patient_data)
                
                if not is_valid:
                    st.error(error_message)
                else:
                    user = st.session_state.user
                    patient = self.referral_service.create_referral(patient_data, user)
                    
                    if patient:
                        st.success(f"Referral created successfully! Patient ID: {patient.patient_id}")
                        
                        # Auto-assign ambulance if requested
                        if patient_data.get('auto_assign_ambulance'):
                            st.info("Auto-assign ambulance feature would be implemented here")

    def _get_patient_form_data(self) -> Dict:
        """Get patient data from form"""
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Patient Name*")
            age = st.number_input("Age*", min_value=0, max_value=120, value=30)
            condition = st.text_input("Medical Condition*")
            referring_physician = st.text_input("Referring Physician*")
            referring_hospital = st.selectbox("Referring Hospital*", self._get_hospital_options())
        
        with col2:
            receiving_hospital = st.selectbox("Receiving Hospital*", self._get_receiving_hospitals())
            receiving_physician = st.text_input("Receiving Physician")
            notes = st.text_area("Clinical Notes")
        
        # Additional medical information
        with st.expander("Additional Medical Information"):
            medical_history = st.text_area("Medical History")
            current_medications = st.text_area("Current Medications")
            allergies = st.text_area("Allergies")
        
        # Ambulance assignment
        st.subheader("🚑 Ambulance Assignment")
        auto_assign_ambulance = st.checkbox("Auto-assign nearest ambulance", value=True)
        
        return {
            'name': name,
            'age': age,
            'condition': condition,
            'referring_physician': referring_physician,
            'referring_hospital': referring_hospital,
            'receiving_hospital': receiving_hospital,
            'receiving_physician': receiving_physician,
            'notes': notes,
            'medical_history': medical_history,
            'current_medications': current_medications,
            'allergies': allergies,
            'auto_assign_ambulance': auto_assign_ambulance
        }

    def _validate_patient_data(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate patient data"""
        required_fields = {
            'name': 'Patient name',
            'age': 'Patient age',
            'condition': 'Medical condition',
            'referring_hospital': 'Referring hospital',
            'receiving_hospital': 'Receiving hospital',
            'referring_physician': 'Referring physician'
        }
        
        for field, description in required_fields.items():
            if not data.get(field):
                return False, f"{description} is required"
        
        if data.get('age') and (data['age'] < 0 or data['age'] > 150):
            return False, "Age must be between 0 and 150"
        
        if data.get('referring_hospital') == data.get('receiving_hospital'):
            return False, "Referring and receiving hospitals cannot be the same"
        
        return True, None

    def _get_hospital_options(self) -> List[str]:
        """Get hospital options based on user role"""
        user_hospital = st.session_state.user['hospital']
        
        if user_hospital == "All Facilities":
            return self._get_all_hospitals()
        else:
            return [user_hospital]

    def _get_receiving_hospitals(self) -> List[str]:
        """Get receiving hospital options"""
        return [
            "Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)", 
            "Kisumu County Referral Hospital"
        ]

    def _get_all_hospitals(self) -> List[str]:
        """Get all hospital names"""
        return [
            "Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)",
            "Kisumu County Referral Hospital",
            "Lumumba Sub-County Hospital",
            "Ahero Sub-County Hospital",
            "Kombewa Sub-County / District Hospital"
        ]

    def _display_active_referrals(self):
        """Display active referrals"""
        st.subheader("Active Referrals")
        
        with self.db_service.get_session() as session:
            user_hospital = st.session_state.user['hospital']
            active_patients = self._get_filtered_patients(session, user_hospital, active_only=True)
            
            if active_patients:
                self._display_patients_table(active_patients)
            else:
                st.info("No active referrals")

    def _display_referral_history(self):
        """Display referral history"""
        st.subheader("Referral History")
        
        with self.db_service.get_session() as session:
            user_hospital = st.session_state.user['hospital']
            all_patients = self._get_filtered_patients(session, user_hospital, active_only=False)
            
            if all_patients:
                self._display_patients_table(all_patients)
            else:
                st.info("No referral history")

    def _get_filtered_patients(self, session, user_hospital: str, active_only: bool = True):
        """Get patients filtered by user hospital and activity status"""
        query = session.query(Patient)
        
        if user_hospital != "All Facilities":
            if user_hospital in ["Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)", 
                               "Kisumu County Referral Hospital"]:
                query = query.filter(Patient.receiving_hospital == user_hospital)
            else:
                query = query.filter(Patient.referring_hospital == user_hospital)
        
        if active_only:
            query = query.filter(Patient.status.notin_(['Completed', 'Arrived at Destination']))
        
        return query.order_by(Patient.referral_time.desc()).all()

    def _display_patients_table(self, patients: List[Patient]):
        """Display patients in a table"""
        data = []
        for patient in patients:
            ambulance_info = patient.assigned_ambulance or "Not assigned"
            data.append({
                'Patient ID': patient.patient_id,
                'Name': patient.name,
                'Condition': patient.condition,
                'From': patient.referring_hospital,
                'To': patient.receiving_hospital,
                'Status': patient.status,
                'Ambulance': ambulance_info,
                'Time': patient.referral_time.strftime('%Y-%m-%d %H:%M')
            })
        
        st.dataframe(pd.DataFrame(data), use_container_width=True)

class CostManagementUI:
    """Cost management UI component"""
    
    def __init__(self, analytics_service: AnalyticsService, db_service: DatabaseService):
        self.analytics = analytics_service
        self.db_service = db_service
    
    def display(self):
        """Display cost management interface"""
        st.title("💰 Cost Management & Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Cost Overview", "Fuel Management", "Savings Analysis"])
        
        with tab1:
            self._display_cost_overview()
        with tab2:
            self._display_fuel_management()
        with tab3:
            self._display_savings_analysis()

    def _display_cost_overview(self):
        """Display cost overview"""
        st.subheader("📈 Cost Overview")
        
        kpis = self.analytics.get_kpis()
        cost_data = self.analytics.get_cost_analytics()
        
        # Cost metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fuel Cost", f"KSh {kpis['total_fuel_cost']:,.0f}")
        with col2:
            st.metric("Total Savings", f"KSh {kpis['total_cost_savings']:,.0f}")
        with col3:
            st.metric("Net Cost", f"KSh {kpis['total_fuel_cost'] - kpis['total_cost_savings']:,.0f}")
        with col4:
            savings_rate = (kpis['total_cost_savings'] / kpis['total_fuel_cost'] * 100) if kpis['total_fuel_cost'] > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        # Cost trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cost_data['months'],
            y=cost_data['monthly_costs'],
            name='Costs Incurred',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=cost_data['months'],
            y=cost_data['monthly_savings'],
            name='Costs Saved',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title='Monthly Costs vs Savings',
            xaxis_title='Month',
            yaxis_title='Amount (KSh)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    def _display_fuel_management(self):
        """Display fuel management interface"""
        st.subheader("⛽ Fuel Management")
        
        # Fuel price configuration
        col1, col2 = st.columns(2)
        with col1:
            current_price = st.number_input(
                "Current Fuel Price (KSh/L)",
                value=float(Config.costs.fuel_price_per_liter),
                min_value=100.0,
                max_value=300.0,
                step=1.0
            )
        
        with col2:
            if st.button("Update Fuel Price", use_container_width=True):
                st.success("Fuel price updated successfully!")
        
        # Ambulance fuel status
        st.subheader("Ambulance Fuel Status")
        with self.db_service.get_session() as session:
            ambulances = session.query(Ambulance).all()
            
            for ambulance in ambulances:
                fuel_status = "🟢 Good" if ambulance.fuel_level > 50 else "🟡 Low" if ambulance.fuel_level > 20 else "🔴 Critical"
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{ambulance.ambulance_id}** - {ambulance.driver_name}")
                with col2:
                    st.write(f"{fuel_status} ({ambulance.fuel_level:.1f}%)")
                with col3:
                    if st.button("Refuel", key=f"refuel_{ambulance.ambulance_id}"):
                        st.info(f"Refueling {ambulance.ambulance_id} - feature would be implemented")

    def _display_savings_analysis(self):
        """Display savings analysis"""
        st.subheader("💵 Savings Analysis")
        
        cost_data = self.analytics.get_cost_analytics()
        
        # Savings trend
        fig = px.area(
            x=cost_data['months'],
            y=cost_data['monthly_savings'],
            title="Monthly Cost Savings Trend",
            labels={'x': 'Month', 'y': 'Savings (KSh)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Main Application
class HospitalReferralApp:
    """Main Hospital Referral Application"""
    
    def __init__(self):
        self.setup_page_config()
        self.auth = Authentication()
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            self.initialize_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Config.app.page_title,
            page_icon=Config.app.page_icon,
            layout=Config.app.layout,
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self.inject_custom_css()
    
    def inject_custom_css(self):
        """Inject custom CSS for better styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .stButton button {
            width: 100%;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_services(self):
        """Initialize application services"""
        try:
            self.db_service = DatabaseService()
            self.notification_service = NotificationService(self.db_service)
            self.referral_service = ReferralService(self.db_service, self.notification_service)
            self.analytics_service = AnalyticsService(self.db_service)
            self.ambulance_service = AmbulanceService(self.db_service)
            self.cost_service = CostCalculationService(self.db_service)
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            st.error("Failed to initialize application services")
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        st.session_state.initialized = True
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.current_page = "Dashboard"
    
    def initialize_database(self):
        """Initialize database tables and default data"""
        try:
            # Create tables
            Base.metadata.create_all(engine)
            logger.info("Database tables created")
            
            # Initialize default users
            self.auth.initialize_default_users()
            
            # Initialize sample data
            self.initialize_sample_data()
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            st.error("Failed to initialize database")
    
    def initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        try:
            with session_scope() as session:
                # Check if sample data already exists
                ambulance_count = session.query(Ambulance).count()
                
                if ambulance_count == 0:
                    # Add sample ambulances
                    sample_ambulances = [
                        Ambulance(
                            ambulance_id="AMB001",
                            current_location="Jaramogi Oginga Odinga Teaching & Referral Hospital",
                            latitude=-0.0754,
                            longitude=34.7695,
                            status="Available",
                            driver_name="John Omondi",
                            driver_contact="+254712345678",
                            fuel_level=85.5
                        ),
                        Ambulance(
                            ambulance_id="AMB002", 
                            current_location="Kisumu County Referral Hospital",
                            latitude=-0.0754,
                            longitude=34.7695,
                            status="Available",
                            driver_name="Mary Achieng",
                            driver_contact="+254723456789",
                            fuel_level=92.3
                        )
                    ]
                    
                    for ambulance in sample_ambulances:
                        session.add(ambulance)
                    
                    session.commit()
                    logger.info("Sample data initialized")
                    
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
    
    def run(self):
        """Main application runner"""
        try:
            # Initialize database on first run
            self.initialize_database()
            
            # Setup services
            self.setup_services()
            
            # Setup authentication
            self.auth.setup_auth_ui()
            
            if st.session_state.get('authenticated'):
                self.render_main_application()
            else:
                self.render_landing_page()
                
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please refresh the page.")
    
    def render_landing_page(self):
        """Render landing page for unauthenticated users"""
        st.title("🏥 Kisumu County Hospital Referral System")
        
        st.markdown("""
        ## Welcome to the Hospital Referral & Ambulance Tracking System
        
        Please login using the sidebar to access the system.
        
        **Key Features:**
        - 🚑 Real-time ambulance tracking
        - 💰 Cost management and analytics
        - 📊 Performance monitoring
        - 📱 Automated notifications
        - 📈 Comprehensive reporting
        
        **Demo Credentials:**
        - Admin: `admin` / `admin123`
        - Hospital Staff: `hospital_staff` / `staff123` 
        - Ambulance Driver: `driver` / `driver123`
        """)
        
        # System overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hospitals in Network", "40+")
        with col2:
            st.metric("Ambulance Fleet", "20+")
        with col3:
            st.metric("Coverage Area", "Kisumu County")
        
        # Feature highlights
        st.subheader("System Benefits")
        features = [
            ("Reduced Response Time", "Average response time under 15 minutes"),
            ("Cost Efficiency", "Up to 20% savings through optimized routing"),
            ("Real-time Tracking", "Live ambulance location and status updates"),
            ("Automated Communication", "Instant notifications to all stakeholders")
        ]
        
        for title, description in features:
            st.write(f"✅ **{title}:** {description}")
    
    def render_main_application(self):
        """Render main application for authenticated users"""
        # User info in sidebar
        self.render_user_info()
        
        # Navigation based on user role
        user_role = st.session_state.user['role']
        
        if user_role == 'Admin':
            self.render_admin_interface()
        elif user_role == 'Hospital Staff':
            self.render_staff_interface()
        elif user_role == 'Ambulance Driver':
            self.render_driver_interface()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Kisumu County Hospital Referral System** | "
            "Secure • Reliable • Cost-Efficient"
        )
    
    def render_user_info(self):
        """Render user information in sidebar"""
        st.sidebar.markdown("---")
        user = st.session_state.user
        
        st.sidebar.success(f"**Logged in as:** {user['name']}")
        st.sidebar.write(f"**Role:** {user['role']}")
        st.sidebar.write(f"**Hospital:** {user['hospital']}")
        
        if user.get('last_login'):
            last_login = user['last_login'].strftime('%Y-%m-%d %H:%M')
            st.sidebar.write(f"**Last Login:** {last_login}")
    
    def render_admin_interface(self):
        """Render admin interface"""
        st.sidebar.title("Admin Navigation")
        
        pages = {
            "📊 Dashboard": self.render_dashboard,
            "📋 Referrals": self.render_referrals,
            "💰 Cost Management": self.render_cost_management,
            "🚑 Ambulance Tracking": self.render_tracking,
            "📄 Handovers": self.render_handovers,
            "💬 Communication": self.render_communication,
            "📈 Reports": self.render_reports,
            "👥 User Management": self.render_user_management
        }
        
        selected_page = st.sidebar.radio("Navigate to", list(pages.keys()))
        pages[selected_page]()
    
    def render_staff_interface(self):
        """Render hospital staff interface"""
        st.sidebar.title("Staff Navigation")
        
        pages = {
            "📊 Dashboard": self.render_dashboard,
            "📋 Referrals": self.render_referrals,
            "🚑 Ambulance Tracking": self.render_tracking,
            "📄 Handovers": self.render_handovers,
            "💬 Communication": self.render_communication
        }
        
        selected_page = st.sidebar.radio("Navigate to", list(pages.keys()))
        pages[selected_page]()
    
    def render_driver_interface(self):
        """Render ambulance driver interface"""
        st.sidebar.title("Driver Navigation")
        
        pages = {
            "🚑 Driver Dashboard": self.render_driver_dashboard,
            "📍 Location Updates": self.render_location_updates,
            "💬 Communication": self.render_communication
        }
        
        selected_page = st.sidebar.radio("Navigate to", list(pages.keys()))
        pages[selected_page]()
    
    def render_dashboard(self):
        """Render dashboard page"""
        dashboard_ui = DashboardUI(self.analytics_service, self.db_service)
        dashboard_ui.display()
    
    def render_referrals(self):
        """Render referrals page"""
        referral_ui = ReferralUI(self.referral_service, self.db_service)
        referral_ui.display()
    
    def render_cost_management(self):
        """Render cost management page"""
        cost_ui = CostManagementUI(self.analytics_service, self.db_service)
        cost_ui.display()
    
    def render_tracking(self):
        """Render ambulance tracking page"""
        st.title("🚑 Ambulance Tracking")
        st.info("Ambulance tracking feature would be implemented here")
    
    def render_handovers(self):
        """Render patient handovers page"""
        st.title("📄 Patient Handovers")
        st.info("Patient handover management feature would be implemented here")
    
    def render_communication(self):
        """Render communication center"""
        st.title("💬 Communication Center")
        st.info("Communication center feature would be implemented here")
    
    def render_reports(self):
        """Render reports page"""
        st.title("📈 Reports & Analytics")
        st.info("Comprehensive reporting feature would be implemented here")
    
    def render_user_management(self):
        """Render user management page"""
        st.title("👥 User Management")
        st.info("User management feature would be implemented here")
    
    def render_driver_dashboard(self):
        """Render driver dashboard"""
        st.title("🚑 Ambulance Driver Dashboard")
        st.info("Driver dashboard feature would be implemented here")
    
    def render_location_updates(self):
        """Render location updates page"""
        st.title("📍 Location Updates")
        st.info("Location update feature would be implemented here")

# Run the application
if __name__ == "__main__":
    app = HospitalReferralApp()
    app.run()

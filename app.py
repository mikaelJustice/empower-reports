# app.py - Enhanced Empower Reports: Cambridge School Report System with Persistent Storage

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import json
import base64
import os
from pathlib import Path
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import shutil
import time

# -------------------------------
# 0. ENHANCED PERSISTENT STORAGE SETUP
# -------------------------------
import os
import shutil
from pathlib import Path
import streamlit as st
import json
from datetime import datetime

# Local development setup
STORAGE_DIR = Path('.')
DB_PATH = Path('empower.db')
BACKUP_DIR = Path('./backups')
BACKUP_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = Path('./uploads')
UPLOADS_DIR.mkdir(exist_ok=True)

EXPORTS_DIR = Path('./exports')
EXPORTS_DIR.mkdir(exist_ok=True)

st.sidebar.info("🔵 Local Storage Mode")

# Store paths in session state for easy access
if 'storage_paths' not in st.session_state:
    st.session_state.storage_paths = {
        'storage_dir': str(STORAGE_DIR),
        'db_path': str(DB_PATH),
        'backup_dir': str(BACKUP_DIR),
        'uploads_dir': str(UPLOADS_DIR),
        'exports_dir': str(EXPORTS_DIR)
    }

# -------------------------------
# BACKUP AND RECOVERY FUNCTIONS
# -------------------------------
def backup_database():
    """Create a backup of database"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"empower_backup_{timestamp}.db"
        
        # Copy database file
        shutil.copy2(DB_PATH, backup_path)
        
        # Log the backup
        with open(BACKUP_DIR / "backup_log.json", "a") as f:
            json.dump({
                "timestamp": timestamp,
                "backup_file": str(backup_path),
                "size": os.path.getsize(backup_path)
            }, f)
            f.write("\n")
        
        return True, f"Backup created: {backup_path}"
    except Exception as e:
        return False, f"Backup failed: {str(e)}"

def restore_database(backup_file):
    """Restore database from a backup file"""
    try:
        # Create a backup of current database before restoring
        backup_database()
        
        # Restore from selected backup
        shutil.copy2(backup_file, DB_PATH)
        return True, "Database restored successfully"
    except Exception as e:
        return False, f"Restore failed: {str(e)}"

def list_backups():
    """List all available backups"""
    try:
        backups = []
        for file in BACKUP_DIR.glob("empower_backup_*.db"):
            backups.append({
                "name": file.name,
                "path": str(file),
                "size": os.path.getsize(file),
                "date": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Sort by date (newest first)
        backups.sort(key=lambda x: x["date"], reverse=True)
        return backups
    except Exception as e:
        st.error(f"Error listing backups: {str(e)}")
        return []

def auto_backup_before_critical_operation(operation_name):
    """Create an automatic backup before critical operations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"auto_backup_{operation_name}_{timestamp}.db"
    
    try:
        shutil.copy2(DB_PATH, backup_path)
        return True
    except Exception as e:
        st.error(f"Auto backup failed: {str(e)}")
        return False

def check_and_create_periodic_backup():
    """Create a backup if it's been more than a day since the last one"""
    try:
        # Check if we need a new backup
        backups = list_backups()
        
        if not backups:
            # No backups exist, create one
            backup_database()
            return
        
        # Check most recent backup
        most_recent = backups[0]
        backup_date = datetime.strptime(most_recent["date"], "%Y-%m-%d %H:%M:%S")
        
        # If more than 24 hours since last backup, create a new one
        if (datetime.now() - backup_date).days >= 1:
            backup_database()
    except Exception as e:
        st.error(f"Error in periodic backup: {str(e)}")

# -------------------------------
# FILE UPLOAD PERSISTENCE
# -------------------------------
def persist_uploaded_file(uploaded_file, subfolder=""):
    """Save uploaded file to persistent storage"""
    try:
        # Create subfolder if specified
        if subfolder:
            target_dir = UPLOADS_DIR / subfolder
            target_dir.mkdir(exist_ok=True)
        else:
            target_dir = UPLOADS_DIR
        
        # Save file
        file_path = target_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def get_persisted_files(subfolder=""):
    """Get list of persisted files"""
    try:
        if subfolder:
            target_dir = UPLOADS_DIR / subfolder
        else:
            target_dir = UPLOADS_DIR
        
        if not target_dir.exists():
            return []
        
        files = []
        for file in target_dir.iterdir():
            if file.is_file():
                files.append({
                    "name": file.name,
                    "path": str(file),
                    "size": os.path.getsize(file),
                    "date": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Sort by date (newest first)
        files.sort(key=lambda x: x["date"], reverse=True)
        return files
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
        return []

# Ensure storage directories exist
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

if not os.path.exists(EXPORTS_DIR):
    os.makedirs(EXPORTS_DIR)

# Check for periodic backup
check_and_create_periodic_backup()

# -------------------------------
# 1. DATABASE & MODELS
# -------------------------------
ENGINE = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})
Base = declarative_base()
Session = sessionmaker(bind=ENGINE)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    role = Column(String)
    password_hash = Column(String)
    subjects_taught = Column(String)
    class_teacher_for = Column(String)
    gender = Column(String)
    phone_number = Column(String)

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    year = Column(Integer)
    class_name = Column(String)
    registration_number = Column(String)
    subjects = Column(String)
    subject_history = Column(Text)
    gender = Column(String)
    enrollment_date = Column(String)

class AcademicTerm(Base):
    __tablename__ = 'academic_terms'
    id = Column(Integer, primary_key=True)
    year = Column(Integer)
    term_number = Column(Integer)
    term_name = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    next_term_begins = Column(String)
    is_active = Column(Boolean, default=False)

# NEW: ComponentMark model to store individual test/paper scores
class ComponentMark(Base):
    __tablename__ = 'component_marks'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    subject = Column(String)
    term_id = Column(Integer, ForeignKey('academic_terms.id'))
    component_type = Column(String)  # 'coursework', 'midterm', 'endterm'
    component_name = Column(String)  # e.g., 'Test 1', 'Paper 1', 'Assignment 1'
    score = Column(Float)
    total = Column(Float)
    submitted_by = Column(Integer, ForeignKey('users.id'))
    submitted_at = Column(String, default=lambda: datetime.now().isoformat())

class Mark(Base):
    __tablename__ = 'marks'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    subject = Column(String)
    term_id = Column(Integer, ForeignKey('academic_terms.id'))
    
    coursework_score = Column(Float)
    coursework_total = Column(Float)
    coursework_out_of_20 = Column(Float)
    
    midterm_score = Column(Float)
    midterm_total = Column(Float)
    midterm_out_of_20 = Column(Float)
    
    endterm_score = Column(Float)
    endterm_total = Column(Float)
    endterm_out_of_60 = Column(Float)
    
    total = Column(Float)
    grade = Column(String)
    comment = Column(Text)
    submitted_by = Column(Integer, ForeignKey('users.id'))
    submitted_at = Column(String, default=lambda: datetime.now().isoformat())

class DisciplineReport(Base):
    __tablename__ = 'discipline_reports'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    reported_by = Column(Integer, ForeignKey('users.id'))
    incident_date = Column(String)
    incident_type = Column(String)
    description = Column(Text)
    action_taken = Column(Text)
    status = Column(String, default="Pending")
    admin_notes = Column(Text)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

class ReportDesign(Base):
    __tablename__ = 'report_design'
    id = Column(Integer, primary_key=True)
    school_name = Column(String, default="EMPOWER INTERNATIONAL ACADEMY")
    school_subtitle = Column(String, default="A Christian Boarding International School")
    school_address = Column(String, default="Nswanjere - Off Kampala-Mityana Road")
    school_po_box = Column(String, default="P.O BOX 1030, Kampala-Uganda")
    school_phone = Column(String, default="")
    school_email = Column(String, default="")
    school_website = Column(String, default="")
    logo_data = Column(Text)
    primary_color = Column(String, default="#8B4513")
    report_footer = Column(Text, default="")

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    action = Column(String)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ClassroomBehavior(Base):
    __tablename__ = 'classroom_behavior'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    term_id = Column(Integer, ForeignKey('academic_terms.id'))
    evaluated_by = Column(Integer, ForeignKey('users.id'))
    
    punctuality = Column(String)  # Excellent, Good, Satisfactory, Cause of Concern
    attendance = Column(String)
    manners = Column(String)
    general_behavior = Column(String)
    organisational_skills = Column(String)
    adherence_to_uniform = Column(String)
    leadership_skills = Column(String)
    commitment_to_school = Column(String)
    cooperation_with_peers = Column(String)
    cooperation_with_staff = Column(String)
    participation_in_lessons = Column(String)
    completion_of_homework = Column(String)
    
    evaluated_at = Column(String, default=lambda: datetime.now().isoformat())

# NEW: StudentDecision model for term 3 decisions
class StudentDecision(Base):
    __tablename__ = 'student_decisions'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    term_id = Column(Integer, ForeignKey('academic_terms.id'))
    decision = Column(String)  # Promoted, Repeated, etc.
    decision_made_by = Column(Integer, ForeignKey('users.id'))
    decision_date = Column(String, default=lambda: datetime.now().isoformat())
    notes = Column(Text)

# NEW: VisitationDay model for parent reports
class VisitationDay(Base):
    __tablename__ = 'visitation_days'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    term_id = Column(Integer, ForeignKey('academic_terms.id'))
    visitation_date = Column(String)
    parent_attended = Column(Boolean, default=False)
    report_given = Column(Boolean, default=False)
    notes = Column(Text)
    created_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(String, default=lambda: datetime.now().isoformat())

# Update database to include new tables if they don't exist
def update_database_schema():
    """Update database schema with new tables and columns"""
    from sqlalchemy import inspect, text
    
    inspector = inspect(ENGINE)
    
    # Create new tables if they don't exist
    if 'visitation_days' not in inspector.get_table_names():
        VisitationDay.__table__.create(ENGINE)
    
    # Check if student_decisions table exists
    if 'student_decisions' not in inspector.get_table_names():
        StudentDecision.__table__.create(ENGINE)
    
    # Check if component_marks table exists
    if 'component_marks' not in inspector.get_table_names():
        ComponentMark.__table__.create(ENGINE)

Base.metadata.create_all(ENGINE)
update_database_schema()

# -------------------------------
# 2. GRADING SYSTEM
# -------------------------------
def get_grade(avg):
    if avg is None or pd.isna(avg):
        return "U"
    avg = float(avg)
    if avg >= 90: return "A*"
    elif avg >= 80: return "A"
    elif avg >= 70: return "B"
    elif avg >= 60: return "C"
    elif avg >= 50: return "D"
    elif avg >= 40: return "E"
    else: return "U"

# -------------------------------
# 3. MARK CONVERSION HELPERS
# -------------------------------
def convert_to_base(score, total, base):
    """Convert a score to a different base"""
    if total == 0 or score is None or total is None:
        return 0
    return round((float(score) / float(total)) * base, 1)

def compute_total(cw_20, mt_20, et_60):
    """Calculate total: CW/20 + MT/20 + ET/60 = /100"""
    total = float(cw_20 or 0) + float(mt_20 or 0) + float(et_60 or 0)
    return round(total, 1)

def compute_total_for_vd(cw_20, mt_20, et_60):
    """Calculate total for VD reports - convert to out of 100 for parents"""
    total = float(cw_20 or 0) + float(mt_20 or 0) + float(et_60 or 0)
    # Convert to out of 100 for VD reports
    return round((total / 100.0) * 100, 1)

# NEW: Function to calculate compiled score from component marks
def calculate_compiled_score(session, student_id, subject, term_id, component_type):
    """Calculate compiled score from component marks"""
    component_marks = pd.read_sql(f"""
        SELECT score, total
        FROM component_marks
        WHERE student_id = {student_id} 
        AND subject = '{subject}' 
        AND term_id = {term_id} 
        AND component_type = '{component_type}'
    """, ENGINE)
    
    if component_marks.empty:
        return 0, 0
    
    total_score = component_marks['score'].sum()
    total_possible = component_marks['total'].sum()
    
    return total_score, total_possible

# NEW: Function to update compiled marks in the main marks table
def update_compiled_marks(session, student_id, subject, term_id):
    """Update the compiled marks in the main marks table"""
    # Get component marks for each type
    cw_score, cw_total = calculate_compiled_score(session, student_id, subject, term_id, 'coursework')
    mt_score, mt_total = calculate_compiled_score(session, student_id, subject, term_id, 'midterm')
    et_score, et_total = calculate_compiled_score(session, student_id, subject, term_id, 'endterm')
    
    # Convert to standardized bases
    cw_out_of_20 = convert_to_base(cw_score, cw_total, 20)
    mt_out_of_20 = convert_to_base(mt_score, mt_total, 20)
    et_out_of_60 = convert_to_base(et_score, et_total, 60)
    
    # Calculate total and grade
    total = compute_total(cw_out_of_20, mt_out_of_20, et_out_of_60)
    grade = get_grade(total)
    
    # Check if a mark record already exists
    existing_mark = session.query(Mark).filter_by(
        student_id=student_id,
        subject=subject,
        term_id=term_id
    ).first()
    
    if existing_mark:
        # Update existing record
        existing_mark.coursework_score = cw_score
        existing_mark.coursework_total = cw_total
        existing_mark.coursework_out_of_20 = cw_out_of_20
        
        existing_mark.midterm_score = mt_score
        existing_mark.midterm_total = mt_total
        existing_mark.midterm_out_of_20 = mt_out_of_20
        
        existing_mark.endterm_score = et_score
        existing_mark.endterm_total = et_total
        existing_mark.endterm_out_of_60 = et_out_of_60
        
        existing_mark.total = total
        existing_mark.grade = grade
        existing_mark.submitted_at = datetime.now().isoformat()
    else:
        # Create new record
        new_mark = Mark(
            student_id=student_id,
            subject=subject,
            term_id=term_id,
            coursework_score=cw_score,
            coursework_total=cw_total,
            coursework_out_of_20=cw_out_of_20,
            midterm_score=mt_score,
            midterm_total=mt_total,
            midterm_out_of_20=mt_out_of_20,
            endterm_score=et_score,
            endterm_total=et_total,
            endterm_out_of_60=et_out_of_60,
            total=total,
            grade=grade
        )
        session.add(new_mark)
    
    session.commit()
    return total, grade

# -------------------------------
# 4. HELPERS
# -------------------------------
def log_audit(session, user_id, action, details=""):
    log = AuditLog(user_id=user_id, action=action, details=details)
    session.add(log)
    session.commit()

def init_report_design():
    session = Session()
    design = session.query(ReportDesign).first()
    if not design:
        design = ReportDesign()
        session.add(design)
        session.commit()
    session.close()

def init_admin():
    session = Session()
    admin = session.query(User).filter_by(email='admin').first()
    if not admin:
        admin = User(
            name='System Admin',
            email='admin',
            role='admin',
            password_hash=hashlib.sha256('admin123'.encode()).hexdigest(),
            subjects_taught='',
            class_teacher_for='',
            gender='',
            phone_number=''
        )
        session.add(admin)
        session.commit()
    session.close()

# Function to download image from URL and convert to base64
def download_logo_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = response.content
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return base64_str
    except Exception as e:
        st.error(f"Error downloading logo: {str(e)}")
        return None

# Performance analysis functions
def calculate_class_performance(session, class_name, term_id):
    """Calculate overall class performance metrics"""
    marks_df = pd.read_sql(f"""
        SELECT s.name, s.class_name, m.subject, m.coursework_out_of_20, 
               m.midterm_out_of_20, m.endterm_out_of_60, m.total, m.grade
        FROM marks m
        JOIN students s ON m.student_id = s.id
        WHERE s.class_name = '{class_name}' AND m.term_id = {term_id}
    """, ENGINE)
    
    if marks_df.empty:
        return None
    
    # Overall metrics
    avg_total = marks_df['total'].mean()
    avg_cw = marks_df['coursework_out_of_20'].mean()
    avg_mt = marks_df['midterm_out_of_20'].mean()
    avg_et = marks_df['endterm_out_of_60'].mean()
    
    # Grade distribution
    grade_counts = marks_df['grade'].value_counts()
    total_students = len(marks_df['name'].unique())
    
    # Performance by subject
    subject_performance = marks_df.groupby('subject').agg({
        'total': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'student_count'}).reset_index()
    
    return {
        'avg_total': avg_total,
        'avg_cw': avg_cw,
        'avg_mt': avg_mt,
        'avg_et': avg_et,
        'grade_distribution': grade_counts,
        'total_students': total_students,
        'subject_performance': subject_performance,
        'detailed_data': marks_df
    }

def find_most_improved_students(session, term_id, subject=None):
    """Find most improved students compared to previous term"""
    current_term = session.query(AcademicTerm).get(term_id)
    if not current_term or current_term.term_number == 1:
        return None
    
    # Get previous term
    prev_term = session.query(AcademicTerm).filter_by(
        year=current_term.year,
        term_number=current_term.term_number - 1
    ).first()
    
    if not prev_term:
        return None
    
    # Get marks for both terms
    current_marks = pd.read_sql(f"""
        SELECT m.student_id, s.name, m.subject, m.total as current_total
        FROM marks m
        JOIN students s ON m.student_id = s.id
        WHERE m.term_id = {term_id}
        {'AND m.subject = \'' + subject + '\'' if subject else ''}
    """, ENGINE)
    
    prev_marks = pd.read_sql(f"""
        SELECT m.student_id, m.total as prev_total
        FROM marks m
        WHERE m.term_id = {prev_term.id}
        {'AND m.subject = \'' + subject + '\'' if subject else ''}
    """, ENGINE)
    
    if current_marks.empty or prev_marks.empty:
        return None
    
    # Merge and calculate improvement
    merged = current_marks.merge(prev_marks, on='student_id', how='inner')
    merged['improvement'] = merged['current_total'] - merged['prev_total']
    merged['improvement_pct'] = (merged['improvement'] / merged['prev_total'] * 100).round(2)
    
    # Filter out students with no previous marks or zero previous total
    merged = merged[(merged['prev_total'] > 0) & (merged['improvement'].notna())]
    
    if merged.empty:
        return None
    
    # Get most improved
    most_improved = merged.nlargest(5, 'improvement_pct')
    
    return {
        'most_improved': most_improved,
        'current_term': current_term.term_name,
        'prev_term': prev_term.term_name,
        'subject': subject or 'All Subjects'
    }

def create_performance_charts(performance_data):
    """Create performance charts using Plotly"""
    charts = {}
    
    if performance_data is None:
        return charts
    
    # 1. Overall Performance Comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=['Coursework', 'Midterm', 'Endterm'],
        y=[performance_data['avg_cw'], performance_data['avg_mt'], performance_data['avg_et']],
        name='Average Scores',
        marker_color=['#3498db', '#e74c3c', '#2ecc71']
    ))
    fig1.update_layout(
        title='Overall Performance by Assessment Type',
        xaxis_title='Assessment Type',
        yaxis_title='Average Score',
        height=400
    )
    charts['overall_performance'] = fig1
    
    # 2. Grade Distribution
    fig2 = go.Figure()
    grades = list(performance_data['grade_distribution'].index)
    counts = list(performance_data['grade_distribution'].values)
    
    fig2.add_trace(go.Bar(
        x=grades,
        y=counts,
        marker_color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#95a5a6']
    ))
    fig2.update_layout(
        title='Grade Distribution',
        xaxis_title='Grade',
        yaxis_title='Number of Students',
        height=400
    )
    charts['grade_distribution'] = fig2
    
    # 3. Subject Performance
    fig3 = go.Figure()
    subjects = performance_data['subject_performance']['subject']
    averages = performance_data['subject_performance']['total']
    
    fig3.add_trace(go.Bar(
        x=subjects,
        y=averages,
        marker_color='#9b59b6'
    ))
    fig3.update_layout(
        title='Performance by Subject',
        xaxis_title='Subject',
        yaxis_title='Average Score',
        height=400
    )
    charts['subject_performance'] = fig3
    
    return charts

def create_improvement_chart(improvement_data):
    """Create improvement chart"""
    if improvement_data is None:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=improvement_data['most_improved']['name'],
        y=improvement_data['most_improved']['improvement_pct'],
        marker_color=['#2ecc71' if x > 0 else '#e74c3c' 
                     for x in improvement_data['most_improved']['improvement_pct']],
        text=improvement_data['most_improved']['improvement_pct'].round(1).astype(str) + '%',
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Most Improved Students - {improvement_data["subject"]}',
        subtitle=f'Comparison: {improvement_data["prev_term"]} → {improvement_data["current_term"]}',
        xaxis_title='Student Name',
        yaxis_title='Improvement (%)',
        height=400
    )
    
    return fig

init_admin()
init_report_design()

# -------------------------------
# 6. IMPROVED PDF GENERATION
# -------------------------------
def generate_pdf_report(student_data, term_data, marks, design, behavior_data=None, decision_data=None, is_vd_report=False):
    """Generate PDF matching image format"""
    buffer = io.BytesIO()
    # Use A4 with smaller margins to fit on one page
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           topMargin=0.2*inch, bottomMargin=0.2*inch,
                           leftMargin=0.4*inch, rightMargin=0.4*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=2,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=1
    )
    
    # Logo and Header
    if design.logo_data:
        try:
            # Handle both base64 and URL cases
            if design.logo_data.startswith('http'):
                # It's a URL, download it
                logo_bytes = requests.get(design.logo_data).content
            else:
                # It's base64 encoded
                logo_bytes = base64.b64decode(design.logo_data)
            
            logo_buffer = io.BytesIO(logo_bytes)
            logo = Image(logo_buffer, width=1.0*inch, height=1.0*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 0.05*inch))
        except Exception as e:
            st.warning(f"Could not load logo in PDF: {str(e)}")
    
    story.append(Paragraph(f"<b>{design.school_name}</b>", title_style))
    if design.school_subtitle:
        story.append(Paragraph(design.school_subtitle, header_style))
    if design.school_address:
        story.append(Paragraph(design.school_address, header_style))
    if design.school_po_box:
        story.append(Paragraph(design.school_po_box, header_style))
    
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<b>END OF TERM {term_data['term_number']} REPORT</b>", title_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Student Info Table - Fixed registration number width
    student_info = [
        ['NAME:', student_data['name'], 'REG. NO:', student_data['registration_number']],
        ['CLASS:', student_data['class_name'], 'YEAR:', f"{term_data['year']} - TERM {term_data['term_number']}"]
    ]
    
    student_table = Table(student_info, colWidths=[0.8*inch, 2.2*inch, 0.8*inch, 2.2*inch])
    student_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3)
    ]))
    
    story.append(student_table)
    story.append(Spacer(1, 0.1*inch))
    
    # Results Table - matching image format with parent-friendly format
    results_data = [['SUBJECTS', 'CW', 'MOT', 'EOT', 'TOTAL', 'GRADE', 'Comment', 'Teacher']]
    
    # Core subjects section
    results_data.append([Paragraph('<b>Core subjects</b>', styles['Normal']), '', '', '', '', '', '', '', ''])
    
    for _, row in marks.iterrows():
        results_data.append([
            row['subject'],
            f"{row['coursework_out_of_20']:.0f}" if row['coursework_out_of_20'] else '-',
            f"{row['midterm_out_of_20']:.0f}" if row['midterm_out_of_20'] else '-',
            f"{row['endterm_out_of_60']:.0f}" if row['endterm_out_of_60'] else '-',
            f"{row['total']:.0f}",
            row['grade'],
            Paragraph(str(row['comment'])[:60] if row['comment'] else '', 
                     ParagraphStyle('Comment', fontSize=7, leading=8)),
            Paragraph(str(row['teacher_name'])[:15], ParagraphStyle('Teacher', fontSize=7))
        ])
    
    results_table = Table(results_data, colWidths=[1.2*inch, 0.4*inch, 0.4*inch, 0.4*inch, 
                                                    0.4*inch, 0.35*inch, 1.5*inch, 0.7*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D3D3D3')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7),
        ('FONTSIZE', (0,1), (-1,-1), 7),
        ('ALIGN', (1,0), (5,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F5F5F5'))
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.1*inch))
    
    # FIXED: Simple average display in a clear table format
    # For VD reports, use the converted out of 100 value
    if is_vd_report:
        overall_avg = compute_total_for_vd(
            marks['coursework_out_of_20'].mean() if not marks.empty else 0,
            marks['midterm_out_of_20'].mean() if not marks.empty else 0,
            marks['endterm_out_of_60'].mean() if not marks.empty else 0
        )
    else:
        overall_avg = marks['total'].mean() if not marks.empty else 0
    
    overall_grade = get_grade(overall_avg)
    
    # Parent-friendly average interpretation
    if overall_grade == "A*":
        avg_comment = "Outstanding Performance"
    elif overall_grade == "A":
        avg_comment = "Excellent Performance"
    elif overall_grade == "B":
        avg_comment = "Good Performance"
    elif overall_grade == "C":
        avg_comment = "Satisfactory Performance"
    elif overall_grade == "D":
        avg_comment = "Needs Improvement"
    elif overall_grade == "E":
        avg_comment = "Poor Performance"
    else:
        avg_comment = "Unsatisfactory Performance"
    
    # Create a simple average table - CLEAR FORMAT
    avg_data = [
        ['AVERAGE', f"{overall_avg:.0f}", overall_grade, avg_comment]
    ]
    
    avg_table = Table(avg_data, colWidths=[1.2*inch, 0.8*inch, 0.6*inch, 2.4*inch])
    avg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D3D3D3')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (1,0), (2,0), 'CENTER'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (3,0), (3,0), 'LEFT'),
        ('VALIGN', (0,0), (-1,0), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('BOTTOMPADDING', (0,0), (-1,0), 5)
    ]))
    
    story.append(avg_table)
    story.append(Spacer(1, 0.1*inch))
    
    # MODIFIED: Only show decision for term 3 and make it part of table
    if term_data['term_number'] == 3 and decision_data:
        decision = decision_data.get('decision', 'Pending')
        
        decision_table_data = [
            ['DECISION', decision, '', '', '', '', '', '', '', '']
        ]
        
        decision_table = Table(decision_table_data, colWidths=[1.2*inch, 0.4*inch, 0.4*inch, 0.4*inch, 
                                                          0.4*inch, 0.35*inch, 1.5*inch, 0.7*inch])
        decision_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#D3D3D3')),
            ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (1,0), 7),
            ('ALIGN', (1,0), (5,0), 'CENTER'),
            ('ALIGN', (0,0), (0,0), 'LEFT'),
            ('VALIGN', (0,0), (1,0), 'MIDDLE'),
            ('TOPPADDING', (0,0), (1,0), 3),
            ('BOTTOMPADDING', (0,0), (1,0), 3)
        ]))
        
        story.append(decision_table)
        story.append(Spacer(1, 0.1*inch))
    
    # Grading scale and key - side by side with smaller size
    grading_info = [
        [
            Paragraph("""<b>Grading Scale</b><br/>
            90-100 = A*<br/>
            80-89 = A<br/>
            70-79 = B<br/>
            60-69 = C<br/>
            50-59 = D<br/>
            40-49 = E<br/>
            0-19 = U""", ParagraphStyle('Grading', fontSize=7, leading=8)),
            
            Paragraph("""<b>KEY</b><br/>
            CW : Coursework<br/>
            MOT : Mid of term Test<br/>
            EOT : End of term Exam<br/>
            GR : Grade""", ParagraphStyle('Key', fontSize=7, leading=8))
        ]
    ]
    
    grading_table = Table(grading_info, colWidths=[2.5*inch, 2.5*inch])
    grading_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5)
    ]))
    
    story.append(grading_table)
    story.append(Spacer(1, 0.05*inch))
    
    # Classroom Behaviour Table - smaller size
    story.append(Paragraph("<b>CLASSROOM BEHAVIOR:</b>", 
                          ParagraphStyle('BehaviorTitle', fontSize=9, fontName='Helvetica-Bold')))
    story.append(Spacer(1, 0.02*inch))
    
    behavior_data_table = [['', 'Excellent', 'Good', 'Satisfactory', 'Concern']]
    
    behavior_items = [
        'Punctuality', 'Attendance', 'Manners',
        'General Behavior', 'Organisation',
        'Adherence to Uniform', 'Leadership',
        'Commitment to School', 'Cooperation with Peers',
        'Cooperation with Staff', 'Participation',
        'Homework Completion'
    ]
    
    # FIXED: Map database field names to display labels
    behavior_field_mapping = {
        'Punctuality': 'punctuality',
        'Attendance': 'attendance',
        'Manners': 'manners',
        'General Behavior': 'general_behavior',
        'Organisation': 'organisational_skills',
        'Adherence to Uniform': 'adherence_to_uniform',
        'Leadership': 'leadership_skills',
        'Commitment to School': 'commitment_to_school',
        'Cooperation with Peers': 'cooperation_with_peers',
        'Cooperation with Staff': 'cooperation_with_staff',
        'Participation': 'participation_in_lessons',
        'Homework Completion': 'completion_of_homework'
    }
    
    # Build behavior table with checkmarks
    for item_label in behavior_items:
        row = [item_label]
        
        if behavior_data:
            # Get the field name from mapping
            field_name = behavior_field_mapping.get(item_label)
            
            if field_name:
                # Get the rating from behavior_data
                rating = behavior_data.get(field_name, '')
                
                # Place checkmark in correct column
                if rating == 'Excellent':
                    row.extend(['✓', '', '', ''])
                elif rating == 'Good':
                    row.extend(['', '✓', '', ''])
                elif rating == 'Satisfactory':
                    row.extend(['', '', '✓', ''])
                elif rating == 'Cause of Concern':
                    row.extend(['', '', '', '✓'])
                else:
                    row.extend(['', '', '', ''])
            else:
                row.extend(['', '', '', ''])
        else:
            # Empty checkboxes if no behavior data
            row.extend(['', '', '', ''])
        
        behavior_data_table.append(row)
    
    behavior_table = Table(behavior_data_table, colWidths=[1.6*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    behavior_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D3D3D3')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    
    story.append(behavior_table)
    story.append(Spacer(1, 0.1*inch))
    
    # Signatures and date - smaller
    next_term_text = term_data.get('next_term_begins', '_______________')
    
    sig_data = [
        [f"The next term begins on: {next_term_text}", ''],
        ['', ''],
        ["Class Teacher's signature: _______________________", "Principal's signature: _______________________"]
    ]
    
    sig_table = Table(sig_data, colWidths=[3.0*inch, 3.0*inch])
    sig_table.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,2), (-1,2), 8)
    ]))
    
    story.append(sig_table)
    
    # Footer
    if design.report_footer:
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph(design.report_footer, header_style))
    
    doc.build(story)
    return buffer.getvalue()

# -------------------------------
# 7. APP SETUP
# -------------------------------
st.set_page_config(page_title="Empower Reports", layout="wide")

storage_status = "🔵 Local Storage Mode"
st.markdown(f"<h1 style='text-align: center; color: #1e3a8a;'>Empower International Academy</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #666; font-size: 14px;'>{storage_status}</p>", unsafe_allow_html=True)

# -------------------------------
# 8. AUTHENTICATION
# -------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.logged_in:
    st.sidebar.header("Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            session = Session()
            user = session.query(User).filter_by(email=username).first()
            
            if user and user.password_hash == hashlib.sha256(password.encode()).hexdigest():
                st.session_state.logged_in = True
                st.session_state.user_role = user.role
                st.session_state.user_id = user.id
                st.session_state.username = user.name
                session.close()
                st.rerun()
            else:
                st.error("Invalid username or password")
                session.close()
    
    st.info("Default login: **admin** / **admin123**")
    st.stop()

# Logout
with st.sidebar:
    st.success(f"Welcome, {st.session_state.username}")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

# Add storage info to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Storage Status")

try:
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
    st.sidebar.metric("Database", f"{db_size:.2f} MB")
except:
    st.sidebar.metric("Database", "Unknown")

st.sidebar.info("📁 Local Storage")

# -------------------------------
# 9. MAIN MENU
# -------------------------------
st.sidebar.title(f"Role: {st.session_state.user_role.title()}")

if st.session_state.user_role == 'admin':
    page = st.sidebar.selectbox("Menu", [
        "Dashboard", 
        "Performance Analytics",  # New menu item
        "Admin Management",
        "Staff Management", 
        "Student Enrollment", 
        "Academic Calendar",
        "Classroom Behavior",
        "Student Decisions",  # New menu item for term 3 decisions
        "Discipline Reports",
        "Generate Reports",
        "Report Design",
        "Data Export",
        "Change Login Details",
        "Visitation Day Management",  # New menu item for VD reports
        "Storage Management"  # New menu item for storage management
    ])
else:
    page = st.sidebar.selectbox("Menu", [
        "Dashboard", 
        "Enter Results", 
        "Classroom Behavior",
        "Student Decisions",  # New menu item for term 3 decisions
        "My Classes", 
        "Discipline Reports",
        "Change Login Details"
    ])

# -------------------------------
# 10. PAGES - Dashboard
# -------------------------------

if page == "Dashboard":
    st.header("📊 Dashboard")
    session = Session()
    
    total_students = session.query(Student).count()
    total_staff = session.query(User).filter(User.role == 'teacher').count()
    total_terms = session.query(AcademicTerm).count()
    
    if total_students == 0 and total_staff == 0 and total_terms == 0 and st.session_state.user_role == 'admin':
        st.info("👋 **Welcome to Empower Reports!** It looks like this is your first time. Let's get you set up!")
        
        with st.expander("📋 Quick Setup Checklist", expanded=True):
            st.markdown("""
            ### Follow these steps to get started:
            
            1. ✅ **Change Admin Password** (Change Login Details)
            2. 📅 **Create Academic Term** (Academic Calendar → Add Term → Set Active)
            3. 🎨 **Customize Reports** (Report Design → Upload logo, change colors)
            4. 👥 **Add Teachers** (Staff Management → Add Staff)
            5. 🎓 **Enroll Students** (Student Enrollment → Enroll Student)
            6. 📝 **Teachers Enter Results** (Teacher accounts → Enter Results)
            7. 📊 **Class Teachers Complete Behavior Reports** (Classroom Behavior)
            8. 📄 **Generate Reports** (Generate Reports)
            9. 📈 **View Performance Analytics** (Performance Analytics)
            """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", total_students)
    with col2:
        st.metric("Total Staff", total_staff)
    with col3:
        st.metric("Pending Discipline Reports", 
                 session.query(DisciplineReport).filter_by(status='Pending').count())
    with col4:
        active_term = session.query(AcademicTerm).filter_by(is_active=True).first()
        st.metric("Active Term", active_term.term_name if active_term else "None")
    
    if st.session_state.user_role == 'admin':
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Total Admins", session.query(User).filter_by(role='admin').count())
        with col6:
            st.metric("Total Teachers", session.query(User).filter_by(role='teacher').count())
    
    # Add storage status section
    st.markdown("---")
    st.subheader("💾 Storage Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
            st.metric("Database Size", f"{db_size:.2f} MB")
        except:
            st.metric("Database Size", "Unknown")
    
    with col2:
        try:
            backup_count = len(list(BACKUP_DIR.glob("empower_backup_*.db")))
            st.metric("Backups", backup_count)
        except:
            st.metric("Backups", "0")
    
    with col3:
        try:
            upload_count = len(list(UPLOADS_DIR.rglob('*')))
            st.metric("Uploaded Files", upload_count)
        except:
            st.metric("Uploaded Files", "0")
    
    with col4:
        st.metric("Storage Mode", "Local")
    
    st.subheader("Recent Activity")
    logs = pd.read_sql("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10", ENGINE)
    if not logs.empty:
        st.dataframe(logs, use_container_width=True)
    else:
        st.info("No activity logs yet")
    session.close()

# NEW PAGE: Storage Management (for admin)
elif page == "Storage Management" and st.session_state.user_role == 'admin':
    st.header("💾 Storage Management")
    
    # Display storage status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
            st.metric("Database Size", f"{db_size:.2f} MB")
        except:
            st.metric("Database Size", "Unknown")
    
    with col2:
        try:
            backup_count = len(list(BACKUP_DIR.glob("empower_backup_*.db")))
            st.metric("Backups", backup_count)
        except:
            st.metric("Backups", "0")
    
    with col3:
        try:
            total_size = sum(f.stat().st_size for f in STORAGE_DIR.rglob('*') if f.is_file()) / (1024 * 1024)  # MB
            st.metric("Total Storage", f"{total_size:.2f} MB")
        except:
            st.metric("Total Storage", "Unknown")
    
    st.markdown("---")
    
    # Backup and restore section
    tab1, tab2, tab3 = st.tabs(["Database Backup", "File Management", "Storage Cleanup"])
    
    with tab1:
        st.subheader("Database Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Create Backup", use_container_width=True):
                success, message = backup_database()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            if st.button("List Backups", use_container_width=True):
                backups = list_backups()
                if backups:
                    st.dataframe(backups)
                else:
                    st.info("No backups found")
        
        # Restore from backup
        backups = list_backups()
        if backups:
            st.markdown("#### Restore from Backup")
            selected_backup = st.selectbox(
                "Select backup to restore:",
                options=backups,
                format_func=lambda x: f"{x['name']} ({x['date']}, {x['size']/1024/1024:.2f} MB)"
            )
            
            if st.button("Restore Selected Backup", type="primary"):
                backup_path = selected_backup['path']
                success, message = restore_database(backup_path)
                if success:
                    st.success(message)
                    st.info("Please refresh the page to see restored data")
                else:
                    st.error(message)
    
    with tab2:
        st.subheader("File Management")
        
        # List all uploaded files
        files = get_persisted_files()
        if files:
            st.dataframe(files)
        else:
            st.info("No files found")
    
    with tab3:
        st.subheader("Storage Cleanup")
        
        # Old backups cleanup
        st.markdown("#### Clean Up Old Backups")
        days_to_keep = st.slider("Days to keep backups:", 7, 90, 30)
        
        if st.button("Clean Old Backups"):
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            deleted_count = 0
            
            for backup in list_backups():
                backup_path = Path(backup["path"])
                if backup_path.stat().st_mtime < cutoff_date:
                    backup_path.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} old backups")
            else:
                st.info("No old backups to delete")

# NEW PAGE: Visitation Day Management (for VD reports)
elif page == "Visitation Day Management" and st.session_state.user_role == 'admin':
    st.header("📅 Visitation Day Management")
    session = Session()
    
    active_term = session.query(AcademicTerm).filter_by(is_active=True).first()
    
    if not active_term:
        st.warning("No active term set. Please contact administrator.")
        session.close()
        st.stop()
    
    st.info(f"📅 Current Term: **{active_term.term_name}**")
    
    # Get all students
    students = pd.read_sql("SELECT * FROM students ORDER BY class_name, name", ENGINE)
    
    if students.empty:
        st.warning("No students enrolled yet")
        session.close()
        st.stop()
    
    # Get visitation days for this term
    visitation_days = pd.read_sql(f"""
        SELECT vd.*, s.name as student_name, s.class_name, s.registration_number
        FROM visitation_days vd
        JOIN students s ON vd.student_id = s.id
        WHERE vd.term_id = {active_term.id}
        ORDER BY vd.visitation_date
    """, ENGINE)
    
    # Get classes
    classes = students['class_name'].unique().tolist()
    
    tab1, tab2 = st.tabs(["Manage Visitation Days", "Generate VD Reports"])
    
    with tab1:
        st.subheader("Manage Visitation Days")
        
        # Add new visitation day
        with st.expander("Add New Visitation Day", expanded=True):
            with st.form("add_visitation_day"):
                visitation_date = st.date_input("Visitation Date")
                
                if st.form_submit_button("Add Visitation Day"):
                    # Create empty visitation records for all students
                    for _, student in students.iterrows():
                        new_visitation = VisitationDay(
                            student_id=student['id'],
                            term_id=active_term.id,
                            visitation_date=str(visitation_date),
                            parent_attended=False,
                            report_given=False,
                            notes="",
                            created_by=st.session_state.user_id
                        )
                        session.add(new_visitation)
                    
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_visitation_day", 
                             f"Added visitation day for {visitation_date}")
                    st.success(f"✅ Visitation day added for {visitation_date}")
                    st.rerun()
        
        # View existing visitation days
        if not visitation_days.empty:
            st.subheader("Existing Visitation Days")
            
            # Group by date and show summary
            visitation_summary = visitation_days.groupby('visitation_date').agg({
                'student_id': 'count',
                'parent_attended': 'sum',
                'report_given': 'sum'
            }).rename(columns={'student_id': 'total_students'}).reset_index()
            
            for _, row in visitation_summary.iterrows():
                with st.expander(f"📅 {row['visitation_date']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Students", row['total_students'])
                    with col2:
                        st.metric("Parents Attended", row['parent_attended'])
                    with col3:
                        st.metric("Reports Given", row['report_given'])
                    
                    # Get students for this visitation day
                    day_students = visitation_days[visitation_days['visitation_date'] == row['visitation_date']]
                    
                    if not day_students.empty:
                        st.dataframe(day_students[['student_name', 'class_name', 'registration_number', 
                                                  'parent_attended', 'report_given', 'notes']], 
                                     use_container_width=True)
    
    with tab2:
        st.subheader("Generate VD Reports")
        
        # Select visitation date
        if not visitation_days.empty:
            visitation_dates = visitation_days['visitation_date'].unique().tolist()
            selected_date = st.selectbox("Select Visitation Date", visitation_dates)
            
            # Get students for this visitation day
            day_students = visitation_days[visitation_days['visitation_date'] == selected_date]
            
            if not day_students.empty:
                st.info(f"Found {len(day_students)} students for {selected_date}")
                
                # Select student
                selected_student = st.selectbox("Select Student", day_students['student_name'].tolist())
                student_id = day_students[day_students['student_name'] == selected_student].iloc[0]['student_id']
                student_data = students[students['id'] == student_id].iloc[0]
                
                # Get marks for this student
                marks = pd.read_sql(f"""
                    SELECT m.*, u.name as teacher_name 
                    FROM marks m
                    JOIN users u ON m.submitted_by = u.id
                    WHERE m.student_id = {student_id} AND m.term_id = {active_term.id}
                    ORDER BY m.subject
                """, ENGINE)
                
                if not marks.empty:
                    st.subheader(f"Results Preview for {selected_student}")
                    
                    display_df = marks[[
                        'subject', 'coursework_out_of_20', 'midterm_out_of_20', 
                        'endterm_out_of_60', 'total', 'grade', 'comment', 'teacher_name'
                    ]].copy()
                    
                    display_df.columns = ['Subject', 'CW/20', 'MOT/20', 'EOT/60', 'Total', 'Grade', 'Comment', 'Teacher']
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show actual total and converted total for VD
                    actual_total = marks['total'].mean()
                    vd_total = compute_total_for_vd(
                        marks['coursework_out_of_20'].mean() if not marks.empty else 0,
                        marks['midterm_out_of_20'].mean() if not marks.empty else 0,
                        marks['endterm_out_of_60'].mean() if not marks.empty else 0
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Actual Total (out of 100)", f"{actual_total:.0f}")
                    with col2:
                        st.metric("VD Total (out of 100)", f"{vd_total:.0f}")
                    
                    if st.button(f"📄 Generate VD Report for {selected_student}", use_container_width=True):
                        try:
                            design = session.query(ReportDesign).first()
                            
                            # Get behavior data
                            behavior_query = f"""
                                SELECT punctuality, attendance, manners, general_behavior, 
                                       organisational_skills, adherence_to_uniform, leadership_skills,
                                       commitment_to_school, cooperation_with_peers, cooperation_with_staff,
                                       participation_in_lessons, completion_of_homework
                                FROM classroom_behavior
                                WHERE student_id = {student_id} AND term_id = {active_term.id}
                            """
                            behavior_result = pd.read_sql(behavior_query, ENGINE)
                            
                            behavior_data = None
                            if not behavior_result.empty:
                                behavior_data = behavior_result.iloc[0].to_dict()
                            
                            # Get decision data (only for term 3)
                            decision_data = None
                            if active_term.term_number == 3:
                                decision_query = f"""
                                    SELECT decision, notes
                                    FROM student_decisions
                                    WHERE student_id = {student_id} AND term_id = {active_term.id}
                                """
                                decision_result = pd.read_sql(decision_query, ENGINE)
                                if not decision_result.empty:
                                    decision_data = decision_result.iloc[0].to_dict()
                            
                            # Generate VD report with is_vd_report=True
                            pdf_data = generate_pdf_report(
                                student_data, active_term, marks, design, 
                                behavior_data, decision_data, is_vd_report=True
                            )
                            
                            st.download_button(
                                "⬇️ Download VD Report",
                                pdf_data,
                                f"{selected_student}_VD_{selected_date}_report.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                            log_audit(session, st.session_state.user_id, "generate_vd_report", 
                                     f"VD Report: {selected_student} - {selected_date}")
                            st.success("✅ VD report generated successfully!")
                            
                            # Update visitation record to mark report as given
                            visitation_record = session.query(VisitationDay).filter_by(
                                student_id=student_id, 
                                visitation_date=str(selected_date)
                            ).first()
                            
                            if visitation_record:
                                visitation_record.report_given = True
                                session.commit()
                        except Exception as e:
                            st.error(f"❌ Error generating VD report: {str(e)}")
                else:
                    st.info(f"No marks found for {selected_student} in {active_term.term_name}")
        else:
            st.info("No visitation days available")
    
    session.close()

# NEW PAGE: Student Decisions (for term 3)
elif page == "Student Decisions":
    st.header("📋 Student Decisions")
    session = Session()
    
    user = session.query(User).get(st.session_state.user_id)
    active_term = session.query(AcademicTerm).filter_by(is_active=True).first()
    
    if not active_term:
        st.warning("No active term set. Please contact administrator.")
        session.close()
        st.stop()
    
    st.info(f"📅 Current Term: **{active_term.term_name}**")
    
    # Only show decisions for term 3
    if active_term.term_number != 3:
        st.warning("Student decisions are only made at the end of Term 3.")
        session.close()
        st.stop()
    
    # For class teachers, show only their class
    if st.session_state.user_role == 'teacher':
        if not user.class_teacher_for:
            st.warning("You are not assigned as a class teacher. Please contact administrator.")
            session.close()
            st.stop()
        
        class_name = user.class_teacher_for
        st.subheader(f"Making Decisions for Students in {class_name}")
        
        # Get students in this class
        students = pd.read_sql(
            f"SELECT id, name, registration_number FROM students WHERE class_name = '{class_name}' ORDER BY name",
            ENGINE
        )
        
        if students.empty:
            st.info(f"No students in {class_name}")
            session.close()
            st.stop()
        
        # Select student
        selected_student = st.selectbox("Select Student", students['name'].tolist())
        student_id = students[students['name'] == selected_student].iloc[0]['id']
        reg_number = students[students['name'] == selected_student].iloc[0]['registration_number']
        
        # Check if decision already exists
        existing_decision = session.query(StudentDecision).filter_by(
            student_id=student_id,
            term_id=active_term.id
        ).first()
        
        if existing_decision:
            st.info(f"✏️ Editing existing decision for {selected_student}")
        
        # Get student's performance for context
        marks = pd.read_sql(f"""
            SELECT m.subject, m.total, m.grade
            FROM marks m
            WHERE m.student_id = {student_id} AND m.term_id = {active_term.id}
            ORDER BY m.subject
        """, ENGINE)
        
        if not marks.empty:
            st.subheader(f"Performance Summary for {selected_student}")
            
            display_df = marks[['subject', 'total', 'grade']].copy()
            display_df.columns = ['Subject', 'Total', 'Grade']
            st.dataframe(display_df, use_container_width=True)
            
            overall_avg = marks['total'].mean()
            overall_grade = get_grade(overall_avg)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Average", f"{overall_avg:.0f}/100")
            with col2:
                st.metric("Overall Grade", overall_grade)
        
        # Decision form
        with st.form("student_decision"):
            st.subheader(f"Decision for: {selected_student} ({reg_number})")
            
            # Get existing decision if available
            default_decision = 0
            if existing_decision:
                default_decision = ["Promoted", "Repeated", "Transferred", "Other"].index(existing_decision.decision) if existing_decision.decision in ["Promoted", "Repeated", "Transferred", "Other"] else 0
            
            decision = st.radio(
                "Decision for this student:", 
                ["Promoted", "Repeated", "Transferred", "Other"], 
                index=default_decision,
                help="This decision will appear on student's end of year report"
            )
            
            notes = st.text_area(
                "Notes (optional):", 
                value=existing_decision.notes if existing_decision else "",
                help="Additional information about this decision"
            )
            
            submit_button = st.form_submit_button("💾 Save Decision", use_container_width=True)
            
            if submit_button:
                # Show what we're about to save
                st.write("**Debug: Attempting to save...**")
                st.write(f"Student ID: {student_id}")
                st.write(f"Term ID: {active_term.id}")
                st.write(f"User ID: {st.session_state.user_id}")
                st.write(f"Ratings collected: {len(decision)} items")
                
                try:
                    if existing_decision:
                        st.write("**Debug: Updating existing record...**")
                        # Update existing decision
                        existing_decision.decision = decision
                        existing_decision.decision_made_by = st.session_state.user_id
                        existing_decision.decision_date = datetime.now().isoformat()
                        existing_decision.notes = notes
                        
                        session.commit()
                        log_audit(session, st.session_state.user_id, "update_decision", 
                                 f"{selected_student} - {active_term.term_name} - {decision}")
                        st.success(f"✅ Decision updated for {selected_student}!")
                        st.balloons()
                        
                    else:
                        st.write("**Debug: Creating new record...**")
                        # Create new decision
                        new_decision = StudentDecision(
                            student_id=student_id,
                            term_id=active_term.id,
                            decision=decision,
                            decision_made_by=st.session_state.user_id,
                            decision_date=datetime.now().isoformat(),
                            notes=notes
                        )
                        
                        session.add(new_decision)
                        session.commit()
                        session.flush()
                        
                        # Verify it was saved
                        verify = session.query(StudentDecision).filter_by(
                            student_id=student_id,
                            term_id=active_term.id
                        ).first()
                        
                        if verify:
                            st.success(f"✅ Decision CREATED for {selected_student}! (ID: {verify.id})")
                        else:
                            st.warning("⚠️ Record created but verification failed")
                        
                        log_audit(session, st.session_state.user_id, "submit_decision", 
                                 f"{selected_student} - {active_term.term_name} - {decision}")
                        st.balloons()
                    
                    # Force rerun after successful save
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    session.rollback()
                    st.error(f"❌ ERROR saving decision!")
                    st.error(f"Error message: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    
                    # Show full traceback
                    import traceback
                    st.code(traceback.format_exc())
                    
                    st.error(f"Debug info:")
                    st.error(f"- Student ID: {student_id}")
                    st.error(f"- Term ID: {active_term.id}")
                    st.error(f"- User ID: {st.session_state.user_id}")
        
        # View existing evaluations
        st.markdown("---")
        st.subheader("Existing Behavior Evaluations for This Class")
        
        # Get all behavior evaluations for this class
        behavior_df = pd.read_sql(f"""
            SELECT cb.*, s.name, s.registration_number
            FROM classroom_behavior cb
            JOIN students s ON cb.student_id = s.id
            WHERE cb.term_id = {active_term.id}
            AND s.class_name = '{class_name}'
            ORDER BY s.name
        """, ENGINE)
        
        if not behavior_df.empty:
            st.success(f"✅ Found {len(behavior_df)} evaluations")
            
            # Summary table
            summary_data = []
            for _, row in behavior_df.iterrows():
                summary_data.append({
                    'Student': row['name'],
                    'Registration': row['registration_number'],
                    'Punctuality': row['punctuality'],
                    'Attendance': row['attendance'],
                    'General Behavior': row['general_behavior'],
                    'Date': row['evaluated_at'][:10] if row['evaluated_at'] else 'N/A'
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
        else:
            st.info("📝 No behavior evaluations completed yet for this class")
    
    # Admin view (simplified)
    elif st.session_state.user_role == 'admin':
        st.info("📋 As an administrator, you can view and edit student decisions for all classes.")
        
        classes = pd.read_sql("SELECT DISTINCT class_name FROM students ORDER BY class_name", ENGINE)
        
        if classes.empty:
            st.info("No classes available")
            session.close()
            st.stop()
        
        selected_class = st.selectbox("Select Class to View", classes['class_name'].tolist())
        
        behavior_df = pd.read_sql(f"""
            SELECT cb.*, s.name, s.registration_number, u.name as evaluator_name
            FROM classroom_behavior cb
            JOIN students s ON cb.student_id = s.id
            JOIN users u ON cb.evaluated_by = u.id
            WHERE cb.term_id = {active_term.id}
            AND s.class_name = '{selected_class}'
            ORDER BY s.name
        """, ENGINE)
        
        if not behavior_df.empty:
            st.success(f"✅ Found {len(behavior_df)} evaluations for {selected_class}")
            st.dataframe(behavior_df[['name', 'registration_number', 'evaluator_name', 
                                         'punctuality', 'attendance', 'general_behavior']], 
                           use_container_width=True)
        else:
            st.info(f"No evaluations yet for {selected_class}")
                
            class_teacher = session.query(User).filter_by(class_teacher_for=selected_class).first()
            if class_teacher:
                st.info(f"📌 Class Teacher: {class_teacher.name} ({class_teacher.email})")
            else:
                st.warning(f"No class teacher assigned to {selected_class}")
    
    session.close()

# NEW PAGE: Performance Analytics
elif page == "Performance Analytics" and st.session_state.user_role == 'admin':
    st.header("📈 Performance Analytics")
    session = Session()
    
    # Get available terms and classes
    terms = pd.read_sql("SELECT * FROM academic_terms ORDER BY year DESC, term_number DESC", ENGINE)
    classes = pd.read_sql("SELECT DISTINCT class_name FROM students ORDER BY class_name", ENGINE)
    subjects = pd.read_sql("SELECT DISTINCT subject FROM marks ORDER BY subject", ENGINE)
    
    if terms.empty or classes.empty:
        st.warning("Need terms and classes with students to analyze performance")
        session.close()
        st.stop()
    
    # Sidebar controls
    st.sidebar.subheader("Analysis Controls")
    
    selected_term = st.sidebar.selectbox("Select Term", terms['term_name'].tolist())
    term_id = terms[terms['term_name'] == selected_term].iloc[0]['id']
    
    analysis_type = st.sidebar.selectbox("Analysis Type", [
        "Class Performance Overview",
        "Most Improved Students",
        "Subject Performance Comparison",
        "Grade Distribution Analysis",
        "Presentation Mode"
    ])
    
    if analysis_type == "Class Performance Overview":
        st.subheader("📊 Class Performance Overview")
        
        selected_class = st.selectbox("Select Class", classes['class_name'].tolist())
        
        # Calculate class performance
        performance = calculate_class_performance(session, selected_class, term_id)
        
        if performance:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Class Average", f"{performance['avg_total']:.1f}/100")
            with col2:
                st.metric("Total Students", performance['total_students'])
            with col3:
                # Calculate pass rate (Grade C and above)
                passing_grades = ['A*', 'A', 'B', 'C']
                pass_count = sum(performance['grade_distribution'].get(grade, 0) for grade in passing_grades)
                pass_rate = (pass_count / performance['total_students'] * 100) if performance['total_students'] > 0 else 0
                st.metric("Pass Rate", f"{pass_rate:.1f}%")
            with col4:
                # Calculate excellence rate (Grade A and above)
                excellence_grades = ['A*', 'A']
                excellence_count = sum(performance['grade_distribution'].get(grade, 0) for grade in excellence_grades)
                excellence_rate = (excellence_count / performance['total_students'] * 100) if performance['total_students'] > 0 else 0
                st.metric("Excellence Rate", f"{excellence_rate:.1f}%")
            
            # Create charts
            charts = create_performance_charts(performance)
            
            # Display charts in tabs
            tab1, tab2, tab3 = st.tabs(["Assessment Performance", "Grade Distribution", "Subject Performance"])
            
            with tab1:
                st.plotly_chart(charts['overall_performance'], use_container_width=True)
            with tab2:
                st.plotly_chart(charts['grade_distribution'], use_container_width=True)
            with tab3:
                st.plotly_chart(charts['subject_performance'], use_container_width=True)
            
            # Detailed data table
            st.subheader("Detailed Performance Data")
            st.dataframe(performance['detailed_data'], use_container_width=True)
            
            # Download option
            csv_data = performance['detailed_data'].to_csv(index=False)
            st.download_button(
                "📥 Download Class Performance Data",
                csv_data,
                f"{selected_class}_performance_{selected_term}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info(f"No performance data available for {selected_class} in {selected_term}")
    
    elif analysis_type == "Most Improved Students":
        st.subheader("🚀 Most Improved Students")
        
        col1, col2 = st.columns(2)
        with col1:
            scope = st.selectbox("Scope", ["All Subjects", "Specific Subject"])
        with col2:
            if scope == "Specific Subject":
                selected_subject = st.selectbox("Select Subject", subjects['subject'].tolist())
            else:
                selected_subject = None
        
        if st.button("Analyze Improvements", use_container_width=True):
            improvement_data = find_most_improved_students(session, term_id, selected_subject)
            
            if improvement_data:
                st.success(f"Analysis Complete: {improvement_data['current_term']}")
                
                # Create improvement chart
                chart = create_improvement_chart(improvement_data)
                st.plotly_chart(chart, use_container_width=True)
                
                # Detailed table
                st.subheader("Top Improved Students")
                # FIXED: Correct syntax for selecting columns
                display_data = improvement_data['most_improved'][['name', 'subject', 'prev_total', 'current_total', 'improvement', 'improvement_pct']].copy()
                display_data.columns = ['Student Name', 'Subject', 'Previous Total', 'Current Total', 'Improvement', 'Improvement %']
                st.dataframe(display_data, use_container_width=True)
                
                # Download option
                csv_data = display_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Improvement Analysis",
                    csv_data,
                    f"improvement_analysis_{improvement_data['subject'].replace(' ', '_')}_{selected_term}.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No improvement data available. This could be because:")
                st.warning("- This is the first term of the academic year")
                st.warning("- No students have marks for consecutive terms")
                st.warning("- No marks available for the selected criteria")
    
    elif analysis_type == "Subject Performance Comparison":
        st.subheader("📚 Subject Performance Comparison")
        
        # Get all subjects performance
        subject_performance = pd.read_sql(f"""
            SELECT m.subject, AVG(m.total) as avg_total, 
                   AVG(m.coursework_out_of_20) as avg_cw,
                   AVG(m.midterm_out_of_20) as avg_mt,
                   AVG(m.endterm_out_of_60) as avg_et,
                   COUNT(DISTINCT m.student_id) as student_count,
                   COUNT(m.id) as total_entries
            FROM marks m
            WHERE m.term_id = {term_id}
            GROUP BY m.subject
            ORDER BY avg_total DESC
        """, ENGINE)
        
        if not subject_performance.empty:
            # Create comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Total Score', 'Coursework Performance', 
                             'Midterm Performance', 'Endterm Performance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average Total
            fig.add_trace(
                go.Bar(x=subject_performance['subject'], y=subject_performance['avg_total'],
                        name='Avg Total', marker_color='#3498db'),
                row=1, col=1
            )
            
            # Coursework
            fig.add_trace(
                go.Bar(x=subject_performance['subject'], y=subject_performance['avg_cw'],
                        name='Avg CW', marker_color='#2ecc71'),
                row=1, col=2
            )
            
            # Midterm
            fig.add_trace(
                go.Bar(x=subject_performance['subject'], y=subject_performance['avg_mt'],
                        name='Avg MT', marker_color='#e74c3c'),
                row=2, col=1
            )
            
            # Endterm
            fig.add_trace(
                go.Bar(x=subject_performance['subject'], y=subject_performance['avg_et'],
                        name='Avg ET', marker_color='#f39c12'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text=f"Subject Performance Comparison - {selected_term}",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Subject Performance Summary")
            display_data = subject_performance.copy()
            display_data.columns = ['Subject', 'Average Total', 'Average CW', 'Average MT', 
                                 'Average ET', 'Number of Students', 'Total Entries']
            st.dataframe(display_data, use_container_width=True)
            
            # Download option
            csv_data = display_data.to_csv(index=False)
            st.download_button(
                "📥 Download Subject Comparison",
                csv_data,
                f"subject_comparison_{selected_term}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No subject performance data available for this term")
    
    elif analysis_type == "Grade Distribution Analysis":
        st.subheader("📊 Grade Distribution Analysis")
        
        # Overall grade distribution
        grade_dist = pd.read_sql(f"""
            SELECT m.grade, COUNT(m.id) as count
            FROM marks m
            WHERE m.term_id = {term_id}
            GROUP BY m.grade
            ORDER BY 
                CASE m.grade
                    WHEN 'A*' THEN 1
                    WHEN 'A' THEN 2
                    WHEN 'B' THEN 3
                    WHEN 'C' THEN 4
                    WHEN 'D' THEN 5
                    WHEN 'E' THEN 6
                    WHEN 'U' THEN 7
                END
        """, ENGINE)
        
        if not grade_dist.empty:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=grade_dist['grade'],
                values=grade_dist['count'],
                hole=0.3,
                marker_colors=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#95a5a6']
            )])
            
            fig.update_layout(
                title=f"Grade Distribution - {selected_term}",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            total_entries = grade_dist['count'].sum()
            st.subheader("Grade Statistics")
            
            for _, row in grade_dist.iterrows():
                percentage = (row['count'] / total_entries * 100).round(1)
                st.metric(f"Grade {row['grade']}", f"{row['count']} entries ({percentage}%)")
        else:
            st.info("No grade distribution data available for this term")
    
    elif analysis_type == "Presentation Mode":
        st.subheader("🎯 Presentation Mode")
        
        st.info("This mode is designed for presenting performance data in meetings or parent-teacher conferences.")
        
        # Auto-refresh interval
        auto_refresh = st.checkbox("Enable Auto-refresh (for live presentations)")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        # Select what to present
        presentation_content = st.selectbox("Select Content to Present", [
            "Class Performance Summary",
            "Top Performing Students",
            "Subject Rankings",
            "Grade Overview"
        ])
        
        if presentation_content == "Class Performance Summary":
            st.subheader("📊 Class Performance Summary")
            
            # Get all classes performance
            all_classes = []
            for class_name in classes['class_name'].tolist():
                perf = calculate_class_performance(session, class_name, term_id)
                if perf:
                    all_classes.append({
                        'Class': class_name,
                        'Average': perf['avg_total'],
                        'Students': perf['total_students'],
                        'Pass Rate': sum(perf['grade_distribution'].get(g, 0) for g in ['A*', 'A', 'B', 'C']) / perf['total_students'] * 100
                    })
            
            if all_classes:
                class_df = pd.DataFrame(all_classes)
                
                # Create ranking chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=class_df['Class'],
                    y=class_df['Average'],
                    text=class_df['Average'].round(1),
                    textposition='auto',
                    marker_color=['#2ecc71' if x >= 70 else '#f39c12' if x >= 50 else '#e74c3c' 
                                 for x in class_df['Average']]
                ))
                
                fig.update_layout(
                    title=f"Class Performance Rankings - {selected_term}",
                    xaxis_title="Class",
                    yaxis_title="Average Score",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                st.dataframe(class_df, use_container_width=True)
        
        elif presentation_content == "Top Performing Students":
            st.subheader("🏆 Top Performing Students")
            
            # Get top students overall
            top_students = pd.read_sql(f"""
                SELECT s.name, s.class_name, AVG(m.total) as avg_total,
                       COUNT(m.id) as subject_count
                FROM marks m
                JOIN students s ON m.student_id = s.id
                WHERE m.term_id = {term_id}
                GROUP BY s.id
                HAVING subject_count >= 3
                ORDER BY avg_total DESC
                LIMIT 20
            """, ENGINE)
            
            if not top_students.empty:
                # Create podium-style visualization
                fig = go.Figure()
                
                colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#3498db'] * 17
                fig.add_trace(go.Bar(
                    x=top_students['name'],
                    y=top_students['avg_total'],
                    text=top_students['avg_total'].round(1),
                    textposition='auto',
                    marker_color=colors[:len(top_students)]
                ))
                
                fig.update_layout(
                    title=f"Top 20 Students - {selected_term}",
                    xaxis_title="Student Name",
                    yaxis_title="Average Score",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                display_data = top_students.copy()
                display_data.columns = ['Student Name', 'Class', 'Average Score', 'Subjects Taken']
                st.dataframe(display_data, use_container_width=True)
        
        elif presentation_content == "Subject Rankings":
            st.subheader("📚 Subject Rankings")
            
            # Subject performance (reuse from earlier)
            subject_perf = pd.read_sql(f"""
                SELECT m.subject, AVG(m.total) as avg_total,
                       COUNT(DISTINCT m.student_id) as student_count
                FROM marks m
                WHERE m.term_id = {term_id}
                GROUP BY m.subject
                ORDER BY avg_total DESC
            """, ENGINE)
            
            if not subject_perf.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=subject_perf['subject'],
                    y=subject_perf['avg_total'],
                    marker_color='#9b59b6'
                ))
                
                fig.update_layout(
                    title=f"Subject Performance Rankings - {selected_term}",
                    xaxis_title="Subject",
                    yaxis_title="Average Score",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif presentation_content == "Grade Overview":
            st.subheader("📊 Grade Overview")
            
            # Create comprehensive grade overview
            grade_overview = pd.read_sql(f"""
                SELECT 
                    s.class_name,
                    m.grade,
                    COUNT(m.id) as count,
                    AVG(m.total) as avg_score
                FROM marks m
                JOIN students s ON m.student_id = s.id
                WHERE m.term_id = {term_id}
                GROUP BY s.class_name, m.grade
                ORDER BY s.class_name, 
                    CASE m.grade
                        WHEN 'A*' THEN 1
                        WHEN 'A' THEN 2
                        WHEN 'B' THEN 3
                        WHEN 'C' THEN 4
                        WHEN 'D' THEN 5
                        WHEN 'E' THEN 6
                        WHEN 'U' THEN 7
                    END
            """, ENGINE)
            
            if not grade_overview.empty:
                # Create heatmap-style visualization
                pivot_data = grade_overview.pivot(index='class_name', columns='grade', values='count').fillna(0)
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlGn',
                    showscale=True
                ))
                
                fig.update_layout(
                    title=f"Grade Distribution by Class - {selected_term}",
                    xaxis_title="Grade",
                    yaxis_title="Class",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh logic
        if auto_refresh:
            st.rerun()
    
    session.close()

elif page == "Enter Results" and st.session_state.user_role == 'teacher':
    st.header("✏️ Enter Results")
    session = Session()
    
    user = session.query(User).get(st.session_state.user_id)
    my_subjects = [s.strip() for s in user.subjects_taught.split(',')] if user.subjects_taught else []
    
    if not my_subjects:
        st.warning("You have no subjects assigned. Please contact administrator.")
        session.close()
        st.stop()
    
    active_term = session.query(AcademicTerm).filter_by(is_active=True).first()
    if not active_term:
        st.warning("No active term set. Please contact administrator.")
        session.close()
        st.stop()
    
    st.info(f"📅 Current Term: **{active_term.term_name}** (ID: {active_term.id})")
    
    # Select Class
    classes = pd.read_sql("SELECT DISTINCT class_name FROM students ORDER BY class_name", ENGINE)
    
    if classes.empty:
        st.info("No students enrolled yet")
        session.close()
        st.stop()
    
    st.markdown("### Step 1: Select Class")
    selected_class = st.selectbox("Class", classes['class_name'].tolist(), label_visibility="collapsed")
    
    # Select what to enter
    st.markdown("### Step 2: Select What to Enter")
    result_types = ["Coursework (CW)", "Mid of Term (MOT)", "End of Term (EOT)", "All Three"]
    selected_type = st.selectbox("Result Type", result_types, label_visibility="collapsed")
    
    # Select Subject
    st.markdown("### Step 3: Select Subject")
    selected_subject = st.selectbox("Subject", my_subjects, label_visibility="collapsed")
    
    # Get students in this class
    students_in_class = pd.read_sql(
        f"SELECT id, name, year, subjects FROM students WHERE class_name = '{selected_class}'",
        ENGINE
    )
    
    def has_subject(subjects_json, subject):
        try:
            subjects = json.loads(subjects_json)
            return subject in subjects
        except:
            return False
    
    students_in_class['has_subject'] = students_in_class['subjects'].apply(
        lambda x: has_subject(x, selected_subject)
    )
    students_with_subject = students_in_class[students_in_class['has_subject']]
    
    if students_with_subject.empty:
        st.warning(f"No students in {selected_class} take {selected_subject}")
        session.close()
        st.stop()
    
    st.markdown("---")
    st.subheader(f"Enter {selected_type} for {selected_class} - {selected_subject}")
    
    # Show existing marks for verification
    with st.expander("📊 View Existing Marks for This Class/Subject"):
        existing_marks_query = f"""
            SELECT s.name, m.coursework_out_of_20, m.midterm_out_of_20, 
                   m.endterm_out_of_60, m.total, m.grade
            FROM marks m
            JOIN students s ON m.student_id = s.id
            WHERE m.term_id = {active_term.id} 
            AND m.subject = '{selected_subject}'
            AND s.class_name = '{selected_class}'
            ORDER BY s.name
        """
        existing_marks_df = pd.read_sql(existing_marks_query, ENGINE)
        
        if not existing_marks_df.empty:
            st.dataframe(existing_marks_df, use_container_width=True)
        else:
            st.info("No marks entered yet for this class/subject combination")
    
    # NEW: Enhanced component marks entry system
    st.markdown("---")
    st.subheader("📝 Enter Component Marks")
    st.info("Enter individual test/paper scores below. The system will automatically compile them into the final scores.")
    
    # Determine which component type to show based on selection
    if selected_type in ["Coursework (CW)", "All Three"]:
        st.markdown("### 📝 Coursework Components")
        
        # Get existing component marks for this student/subject/term
        student_name = st.selectbox("Select Student", students_with_subject['name'].tolist())
        student_id = int(students_with_subject[students_with_subject['name'] == student_name].iloc[0]['id'])
        
        # Get existing component marks for coursework
        cw_components = pd.read_sql(f"""
            SELECT id, component_name, score, total
            FROM component_marks
            WHERE student_id = {student_id} 
            AND subject = '{selected_subject}' 
            AND term_id = {active_term.id}
            AND component_type = 'coursework'
            ORDER BY component_name
        """, ENGINE)
        
        # Display existing components
        if not cw_components.empty:
            st.subheader("Existing Coursework Components")
            st.dataframe(cw_components, use_container_width=True)
        
        # Add new component
        with st.form("add_cw_component"):
            st.markdown("#### Add New Coursework Component")
            component_name = st.text_input("Component Name (e.g., Test 1, Assignment 1)")
            score = st.number_input("Score", 0.0, 1000.0, step=0.5)
            total = st.number_input("Total Marks", 0.1, 1000.0, step=0.5)
            
            if st.form_submit_button("Add Coursework Component"):
                                if component_name and total > 0:
                                        new_component = ComponentMark(
                        student_id=student_id,
                        subject=selected_subject,
                        term_id=active_term.id,
                        component_type='coursework',
                        component_name=component_name,
                        score=score,
                        total=total,
                        submitted_by=st.session_state.user_id
                    )
                    session.add(new_component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_cw_component", 
                             f"{student_name} - {selected_subject} - {component_name}")
                    st.success(f"✅ Coursework component added: {component_name}")
                    st.rerun()
                else:
                    st.error("Please provide a component name and total marks")
        
        # Delete component
        if not cw_components.empty:
            st.markdown("#### Delete Coursework Component")
            component_to_delete = st.selectbox("Select Component to Delete", 
                                           cw_components['id'].tolist(), 
                                           format_func=lambda x: cw_components[cw_components['id']==x]['component_name'].iloc[0])
            
            if st.button("Delete Selected Component", key="delete_cw"):
                component = session.query(ComponentMark).get(component_to_delete)
                if component:
                    session.delete(component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "delete_cw_component", 
                             f"{student_name} - {selected_subject} - {component.component_name}")
                    st.success(f"✅ Component deleted: {component.component_name}")
                    st.rerun()
    
    if selected_type in ["Mid of Term (MOT)", "All Three"]:
        st.markdown("### 📋 Mid of Term Components")
        
        # Get existing component marks for this student/subject/term
        student_name = st.selectbox("Select Student", students_with_subject['name'].tolist(), key="mt_student")
        student_id = int(students_with_subject[students_with_subject['name'] == student_name].iloc[0]['id'])
        
        # Get existing component marks for mid-term
        mt_components = pd.read_sql(f"""
            SELECT id, component_name, score, total
            FROM component_marks
            WHERE student_id = {student_id} 
            AND subject = '{selected_subject}' 
            AND term_id = {active_term.id}
            AND component_type = 'midterm'
            ORDER BY component_name
        """, ENGINE)
        
        # Display existing components
        if not mt_components.empty:
            st.subheader("Existing Mid-term Components")
            st.dataframe(mt_components, use_container_width=True)
        
        # Add new component
        with st.form("add_mt_component"):
            st.markdown("#### Add New Mid-term Component")
            component_name = st.text_input("Component Name (e.g., Paper 1, Paper 2)", key="mt_component_name")
            score = st.number_input("Score", 0.0, 1000.0, step=0.5, key="mt_score")
            total = st.number_input("Total Marks", 0.1, 1000.0, step=0.5, key="mt_total")
            
            if st.form_submit_button("Add Mid-term Component", key="add_mt"):
                if component_name and total > 0:
                    new_component = ComponentMark(
                        student_id=student_id,
                        subject=selected_subject,
                        term_id=active_term.id,
                        component_type='midterm',
                        component_name=component_name,
                        score=score,
                        total=total,
                        submitted_by=st.session_state.user_id
                    )
                    session.add(new_component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_mt_component", 
                             f"{student_name} - {selected_subject} - {component_name}")
                    st.success(f"✅ Mid-term component added: {component_name}")
                    st.rerun()
                else:
                    st.error("Please provide a component name and total marks")
        
        # Delete component
        if not mt_components.empty:
            st.markdown("#### Delete Mid-term Component")
            component_to_delete = st.selectbox("Select Component to Delete", 
                                           mt_components['id'].tolist(), 
                                           format_func=lambda x: mt_components[mt_components['id']==x]['component_name'].iloc[0],
                                           key="mt_delete")
            
            if st.button("Delete Selected Component", key="delete_mt"):
                component = session.query(ComponentMark).get(component_to_delete)
                if component:
                    session.delete(component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "delete_mt_component", 
                             f"{student_name} - {selected_subject} - {component.component_name}")
                    st.success(f"✅ Component deleted: {component.component_name}")
                    st.rerun()
    
    if selected_type in ["End of Term (EOT)", "All Three"]:
        st.markdown("### 📄 End of Term Components")
        
        # Get existing component marks for this student/subject/term
        student_name = st.selectbox("Select Student", students_with_subject['name'].tolist(), key="et_student")
        student_id = int(students_with_subject[students_with_subject['name'] == student_name].iloc[0]['id'])
        
        # Get existing component marks for end-term
        et_components = pd.read_sql(f"""
            SELECT id, component_name, score, total
            FROM component_marks
            WHERE student_id = {student_id} 
            AND subject = '{selected_subject}' 
            AND term_id = {active_term.id}
            AND component_type = 'endterm'
            ORDER BY component_name
        """, ENGINE)
        
        # Display existing components
        if not et_components.empty:
            st.subheader("Existing End-term Components")
            st.dataframe(et_components, use_container_width=True)
        
        # Add new component
        with st.form("add_et_component"):
            st.markdown("#### Add New End-term Component")
            component_name = st.text_input("Component Name (e.g., Paper 1, Paper 2)", key="et_component_name")
            score = st.number_input("Score", 0.0, 1000.0, step=0.5, key="et_score")
            total = st.number_input("Total Marks", 0.1, 1000.0, step=0.5, key="et_total")
            
            if st.form_submit_button("Add End-term Component", key="add_et"):
                if component_name and total > 0:
                    new_component = ComponentMark(
                        student_id=student_id,
                        subject=selected_subject,
                        term_id=active_term.id,
                        component_type='endterm',
                        component_name=component_name,
                        score=score,
                        total=total,
                        submitted_by=st.session_state.user_id
                    )
                    session.add(new_component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_et_component", 
                             f"{student_name} - {selected_subject} - {component_name}")
                    st.success(f"✅ End-term component added: {component_name}")
                    st.rerun()
                else:
                    st.error("Please provide a component name and total marks")
        
        # Delete component
        if not et_components.empty:
            st.markdown("#### Delete End-term Component")
            component_to_delete = st.selectbox("Select Component to Delete", 
                                           et_components['id'].tolist(), 
                                           format_func=lambda x: et_components[et_components['id']==x]['component_name'].iloc[0],
                                           key="et_delete")
            
            if st.button("Delete Selected Component", key="delete_et"):
                component = session.query(ComponentMark).get(component_to_delete)
                if component:
                    session.delete(component)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "delete_et_component", 
                             f"{student_name} - {selected_subject} - {component.component_name}")
                    st.success(f"✅ Component deleted: {component.component_name}")
                    st.rerun()
    
    # Compile and update marks
    st.markdown("---")
    st.subheader("🔄 Compile and Update Final Marks")
    
    # Get students with components
    students_with_components = pd.read_sql(f"""
        SELECT DISTINCT s.id, s.name
        FROM students s
        JOIN component_marks cm ON s.id = cm.student_id
        WHERE s.class_name = '{selected_class}'
        AND s.id IN (
            SELECT student_id FROM component_marks 
            WHERE subject = '{selected_subject}' 
            AND term_id = {active_term.id}
        )
        ORDER BY s.name
    """, ENGINE)
    
    if students_with_components.empty:
        st.info("No students with component marks found. Add component marks first.")
    else:
        selected_student = st.selectbox("Select Student to Compile", students_with_components['name'].tolist())
        student_id = int(students_with_components[students_with_components['name'] == selected_student].iloc[0]['id'])
        
        # Show current compiled scores
        cw_score, cw_total = calculate_compiled_score(session, student_id, selected_subject, active_term.id, 'coursework')
        mt_score, mt_total = calculate_compiled_score(session, student_id, selected_subject, active_term.id, 'midterm')
        et_score, et_total = calculate_compiled_score(session, student_id, selected_subject, active_term.id, 'endterm')
        
        # Convert to standardized bases
        cw_out_of_20 = convert_to_base(cw_score, cw_total, 20)
        mt_out_of_20 = convert_to_base(mt_score, mt_total, 20)
        et_out_of_60 = convert_to_base(et_score, et_total, 60)
        
        # Calculate total and grade
        total = compute_total(cw_out_of_20, mt_out_of_20, et_out_of_60)
        grade = get_grade(total)
        
        st.subheader(f"Compiled Scores for {selected_student}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coursework (out of 20)", f"{cw_out_of_20:.1f}")
            st.write(f"Based on {cw_score:.1f}/{cw_total:.1f}")
        with col2:
            st.metric("Mid-term (out of 20)", f"{mt_out_of_20:.1f}")
            st.write(f"Based on {mt_score:.1f}/{mt_total:.1f}")
        with col3:
            st.metric("End-term (out of 60)", f"{et_out_of_60:.1f}")
            st.write(f"Based on {et_score:.1f}/{et_total:.1f}")
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Total (out of 100)", f"{total:.1f}")
        with col5:
            st.metric("Grade", grade)
        
        # Update button
        if st.button(f"💾 Update Final Marks for {selected_student}", use_container_width=True):
            try:
                total, grade = update_compiled_marks(session, student_id, selected_subject, active_term.id)
                log_audit(session, st.session_state.user_id, "update_compiled_marks", 
                         f"{selected_student} - {selected_subject} - {active_term.term_name}")
                st.success(f"✅ Final marks updated for {selected_student}!")
                st.success(f"📊 Total: {total:.1f}/100 | Grade: {grade}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error updating marks: {str(e)}")
    
    session.close()

# Add remaining pages after "Enter Results"

elif page == "Admin Management" and st.session_state.user_role == 'admin':
    st.header("👑 Admin Management")
    session = Session()

    tab1, tab2 = st.tabs(["View Admins", "Add Admin"])
    
    with tab1:
        st.subheader("All Administrators")
        admins_df = pd.read_sql("""
            SELECT id, name, gender, phone_number, email 
            FROM users WHERE role = 'admin'
        """, ENGINE)
        
        if not admins_df.empty:
            st.dataframe(admins_df, use_container_width=True)
            
            with st.expander("✏️ Edit Admin"):
                admin_id = st.selectbox("Select Admin to Edit", 
                                       admins_df['id'].tolist(), 
                                       format_func=lambda x: admins_df[admins_df['id']==x]['name'].iloc[0])
                
                admin = session.query(User).get(admin_id)
                
                with st.form("edit_admin"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Name", value=admin.name)
                        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                            index=["Male", "Female", "Other"].index(admin.gender) if admin.gender else 0)
                    with col2:
                        phone = st.text_input("Phone Number", value=admin.phone_number or "")
                        email = st.text_input("Email", value=admin.email)
                    
                    if st.form_submit_button("Update Admin"):
                        admin.name = name
                        admin.gender = gender
                        admin.phone_number = phone
                        admin.email = email
                        session.commit()
                        log_audit(session, st.session_state.user_id, "edit_admin", f"Updated {name}")
                        st.success("✅ Admin updated successfully!")
                        st.rerun()
        else:
            st.info("No other admins")
    
    with tab2:
        st.subheader("Add New Administrator")
        with st.form("add_admin"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name*")
                gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                phone = st.text_input("Phone Number*")
            with col2:
                email = st.text_input("Email/Username*")
                password = st.text_input("Password*", type="password")
                confirm_pass = st.text_input("Confirm Password*", type="password")
            
            admin_title = st.text_input("Title (e.g., Principal, DOA, Facilitator)")
            
            if st.form_submit_button("Add Admin"):
                if not all([name, gender, phone, email, password]):
                    st.error("Please fill all required fields marked with *")
                elif password != confirm_pass:
                    st.error("Passwords don't match")
                elif session.query(User).filter_by(email=email).first():
                    st.error("Email already exists")
                else:
                    new_admin = User(
                        name=f"{name} ({admin_title})" if admin_title else name,
                        gender=gender,
                        phone_number=phone,
                        email=email,
                        role='admin',
                        password_hash=hashlib.sha256(password.encode()).hexdigest(),
                        subjects_taught='',
                        class_teacher_for=''
                    )
                    session.add(new_admin)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_admin", f"Added {name}")
                    st.success(f"✅ Admin added: {email} / {password}")
                    st.rerun()
    
    session.close()

elif page == "Staff Management" and st.session_state.user_role == 'admin':
    st.header("👥 Staff Management")
    session = Session()

    tab1, tab2 = st.tabs(["View Staff", "Add Staff"])
    
    with tab1:
        st.subheader("All Staff Members")
        staff_df = pd.read_sql("""
            SELECT id, name, gender, role, phone_number, email, subjects_taught, class_teacher_for 
            FROM users WHERE role = 'teacher'
        """, ENGINE)
        
        if not staff_df.empty:
            st.dataframe(staff_df, use_container_width=True)
            
            with st.expander("✏️ Edit or Delete Staff"):
                staff_id = st.selectbox("Select Staff to Edit", 
                                       staff_df['id'].tolist(), 
                                       format_func=lambda x: staff_df[staff_df['id']==x]['name'].iloc[0])
                
                staff = session.query(User).get(staff_id)
                
                with st.form("edit_staff"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Name", value=staff.name)
                        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                            index=["Male", "Female", "Other"].index(staff.gender) if staff.gender else 0)
                        phone = st.text_input("Phone Number", value=staff.phone_number or "")
                    with col2:
                        email = st.text_input("Email", value=staff.email)
                        subjects = st.text_input("Subjects Taught", value=staff.subjects_taught or "")
                    
                    class_for = st.text_input("Class Teacher For", value=staff.class_teacher_for or "")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.form_submit_button("💾 Update Staff", use_container_width=True):
                            staff.name = name
                            staff.gender = gender
                            staff.phone_number = phone
                            staff.email = email
                            staff.subjects_taught = subjects
                            staff.class_teacher_for = class_for
                            session.commit()
                            log_audit(session, st.session_state.user_id, "edit_staff", f"Updated {name}")
                            st.success("✅ Staff updated successfully!")
                            st.rerun()
                    
                    with col4:
                        if st.form_submit_button("🗑️ Delete Staff", type="primary", use_container_width=True):
                            # First check if staff has any marks
                            marks_count = session.query(Mark).filter_by(submitted_by=staff_id).count()
                            if marks_count > 0:
                                st.error(f"❌ Cannot delete: Staff has {marks_count} mark records. Delete marks first.")
                            else:
                                staff_name = staff.name
                                session.delete(staff)
                                session.commit()
                                log_audit(session, st.session_state.user_id, "delete_staff", f"Deleted {staff_name}")
                                st.success(f"✅ Staff {staff_name} deleted successfully!")
                                st.rerun()
        else:
            st.info("No staff members yet")
    
    with tab2:
        st.subheader("Add New Staff Member")
        with st.form("add_staff"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name*")
                gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                phone = st.text_input("Phone Number*")
            with col2:
                email = st.text_input("Email/Username*")
                password = st.text_input("Password*", type="password")
                confirm_pass = st.text_input("Confirm Password*", type="password")
            
            subjects = st.text_input("Subjects Taught (comma-separated)")
            class_for = st.text_input("Class Teacher For (e.g., Year 8A)")
            
            if st.form_submit_button("Add Staff"):
                if not all([name, gender, phone, email, password]):
                    st.error("Please fill all required fields marked with *")
                elif password != confirm_pass:
                    st.error("Passwords don't match")
                elif session.query(User).filter_by(email=email).first():
                    st.error("Email already exists")
                else:
                    new_staff = User(
                        name=name,
                        gender=gender,
                        phone_number=phone,
                        email=email,
                        role='teacher',
                        password_hash=hashlib.sha256(password.encode()).hexdigest(),
                        subjects_taught=subjects,
                        class_teacher_for=class_for
                    )
                    session.add(new_staff)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_staff", f"Added {name}")
                    st.success(f"✅ Staff added: {email} / {password}")
                    st.rerun()
    
    session.close()

elif page == "Student Enrollment" and st.session_state.user_role == 'admin':
    st.header("🎓 Student Enrollment")
    session = Session()

    tab1, tab2 = st.tabs(["View Students", "Enroll Student"])
    
    with tab1:
        st.subheader("All Students")
        students_df = pd.read_sql("SELECT * FROM students", ENGINE)
        
        if not students_df.empty:
            st.dataframe(students_df, use_container_width=True)
            
            # Edit/Delete Student
            with st.expander("✏️ Edit or Delete Student"):
                student_id = st.selectbox("Select Student", 
                                         students_df['id'].tolist(), 
                                         format_func=lambda x: students_df[students_df['id']==x]['name'].iloc[0])
                
                student = session.query(Student).get(student_id)
                
                # Parse existing subjects
                try:
                    current_subjects = list(json.loads(student.subjects).keys())
                except:
                    current_subjects = []
                
                with st.form("edit_student"):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input("Student Name", value=student.name)
                        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                            index=["Male", "Female", "Other"].index(student.gender) if student.gender else 0)
                        year = st.selectbox("Year", range(7, 14), index=student.year - 7 if student.year else 0)
                    with col2:
                        reg_number = st.text_input("Registration Number", value=student.registration_number)
                        class_name = st.text_input("Class", value=student.class_name)
                        enrollment_date = st.date_input("Enrollment Date", 
                                                       value=pd.to_datetime(student.enrollment_date) if student.enrollment_date else datetime.now())
                    
                    # Subject selection
                    available_subjects = [
                        "Mathematics", "English", "Physics", "Chemistry", "Biology",
                        "History", "Geography", "Business Studies", "Economics",
                        "Computer Science", "ICT", "Art", "Physical Education", "ART",
                        "English First Language", "FRENCH", "GEOGRAPHY", "History",
                        "Information Communication Technology", "MUSIC", "Physical Education",
                        "SCIENCE"
                    ]
                    
                    selected_subjects = st.multiselect("Subjects", available_subjects, default=current_subjects)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.form_submit_button("💾 Update Student", use_container_width=True):
                            if not all([name, gender, class_name, reg_number]) or not selected_subjects:
                                st.error("Please fill all required fields and select at least one subject")
                            else:
                                subjects_json = json.dumps({subj: "Active" for subj in selected_subjects})
                                
                                student.name = name
                                student.gender = gender
                                student.year = year
                                student.class_name = class_name
                                student.registration_number = reg_number
                                student.subjects = subjects_json
                                student.enrollment_date = str(enrollment_date)
                                
                                session.commit()
                                log_audit(session, st.session_state.user_id, "edit_student", f"Updated {name}")
                                st.success("✅ Student updated successfully!")
                                st.rerun()
                    
                    with col4:
                        if st.form_submit_button("🗑️ Delete Student", type="primary", use_container_width=True):
                            # First check if student has any marks
                            marks_count = session.query(Mark).filter_by(student_id=student_id).count()
                            if marks_count > 0:
                                st.error(f"❌ Cannot delete: Student has {marks_count} mark records. Delete marks first.")
                            else:
                                student_name = student.name
                                session.delete(student)
                                session.commit()
                                log_audit(session, st.session_state.user_id, "delete_student", f"Deleted {student_name}")
                                st.success(f"✅ Student {student_name} deleted successfully!")
                                st.rerun()
        else:
            st.info("No students enrolled yet")
    
    with tab2:
        st.subheader("Enroll New Student")
        with st.form("enroll_student"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Student Name*")
                gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                year = st.selectbox("Year*", range(7, 14))
                reg_number = st.text_input("Registration Number*")
            with col2:
                class_name = st.text_input("Class (e.g., Year 8 South)*")
                enrollment_date = st.date_input("Enrollment Date")
            
            st.write("**Select Subjects**")
            available_subjects = [
                "Mathematics", "English", "Physics", "Chemistry", "Biology",
                "History", "Geography", "Business Studies", "Economics",
                "Computer Science", "ICT", "Art", "Physical Education", "ART",
                "English First Language", "FRENCH", "GEOGRAPHY", "History",
                "Information Communication Technology", "MUSIC", "Physical Education",
                "SCIENCE"
            ]
            
            selected_subjects = st.multiselect("Subjects*", available_subjects)
            
            if st.form_submit_button("Enroll Student"):
                if not all([name, gender, class_name, reg_number]) or not selected_subjects:
                    st.error("Please fill all required fields and select at least one subject")
                else:
                    subjects_json = json.dumps({subj: "Active" for subj in selected_subjects})
                    
                    new_student = Student(
                        name=name,
                        gender=gender,
                        year=year,
                        class_name=class_name,
                        registration_number=reg_number,
                        subjects=subjects_json,
                        subject_history=f"Enrolled: {enrollment_date}",
                        enrollment_date=str(enrollment_date)
                    )
                    session.add(new_student)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "enroll_student", f"Enrolled {name}")
                    st.success(f"✅ Student enrolled: {name}")
                    st.rerun()
    
    session.close()

elif page == "Academic Calendar" and st.session_state.user_role == 'admin':
    st.header("📅 Academic Calendar")
    session = Session()

    tab1, tab2 = st.tabs(["View Terms", "Add Term"])
    
    with tab1:
        st.subheader("All Academic Terms")
        terms_df = pd.read_sql("SELECT * FROM academic_terms ORDER BY year DESC, term_number DESC", ENGINE)
        
        
        if not terms_df.empty:
            for _, term in terms_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                with col1:
                    if term['is_active']:
                        st.success(f"✅ **{term['term_name']}** (ACTIVE)")
                    else:
                        st.info(f"📅 {term['term_name']}")
                with col2:
                    st.write(f"Start: {term['start_date']}")
                with col3:
                    st.write(f"End: {term['end_date']}")
                with col4:
                    st.write(f"Next: {term['next_term_begins'] or 'N/A'}")
                with col5:
                    if not term['is_active']:
                        if st.button(f"Set Active", key=f"activate_{term['id']}"):
                            session.query(AcademicTerm).update({AcademicTerm.is_active: False})
                            session.query(AcademicTerm).filter_by(id=term['id']).update({AcademicTerm.is_active: True})
                            session.commit()
                            log_audit(session, st.session_state.user_id, "set_active_term", term['term_name'])
                            st.success(f"✅ Active term set to: {term['term_name']}")
                            st.rerun()
        else:
            st.info("No academic terms created yet")
    
    with tab2:
        st.subheader("Add New Term")
        with st.form("add_term"):
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Year", min_value=2020, max_value=2050, value=2025)
                term_num = st.selectbox("Term Number", [1, 2, 3])
            with col2:
                start = st.date_input("Start Date")
                end = st.date_input("End Date")
            
            next_term = st.date_input("Next Term Begins")
            
            if st.form_submit_button("Add Term"):
                term_name = f"{year} Term {term_num}"
                existing = session.query(AcademicTerm).filter_by(year=year, term_number=term_num).first()
                
                if existing:
                    st.error("This term already exists!")
                else:
                    new_term = AcademicTerm(
                        year=year,
                        term_number=term_num,
                        term_name=term_name,
                        start_date=str(start),
                        end_date=str(end),
                        next_term_begins=str(next_term),
                        is_active=False
                    )
                    session.add(new_term)
                    session.commit()
                    log_audit(session, st.session_state.user_id, "add_term", term_name)
                    st.success(f"✅ Term added: {term_name}")
                    st.rerun()
    
    session.close()

elif page == "Classroom Behavior":
    st.header("📝 Classroom Behavior Evaluation")
    session = Session()
    
    user = session.query(User).get(st.session_state.user_id)
    active_term = session.query(AcademicTerm).filter_by(is_active=True).first()
    
    if not active_term:
        st.warning("No active term set. Please contact administrator.")
        session.close()
        st.stop()
    
    st.info(f"📅 Current Term: **{active_term.term_name}** (ID: {active_term.id})")
    
    # For class teachers, show behavior evaluation form
    if st.session_state.user_role == 'teacher':
        if not user.class_teacher_for:
            st.warning("You are not assigned as a class teacher. Please contact administrator.")
            session.close()
            st.stop()
        
        class_name = user.class_teacher_for
        st.subheader(f"Evaluate Behavior for Students in {class_name}")
        
        # Get students in this class
        students = pd.read_sql(
            f"SELECT id, name, registration_number FROM students WHERE class_name = '{class_name}' ORDER BY name",
            ENGINE
        )
        
        if students.empty:
            st.info(f"No students in {class_name}")
            session.close()
            st.stop()
        
        # Select student to evaluate
        selected_student = st.selectbox("Select Student to Evaluate", students['name'].tolist())
        student_id = int(students[students['name'] == selected_student].iloc[0]['id'])
        reg_number = students[students['name'] == selected_student].iloc[0]['registration_number']
        
        # Check if behavior evaluation already exists
        existing_behavior = session.query(ClassroomBehavior).filter_by(
            student_id=student_id,
            term_id=active_term.id
        ).first()
        
        if existing_behavior:
            st.success(f"✏️ Found existing evaluation - You can update it below")
        else:
            st.info(f"📝 No existing evaluation found - Creating new evaluation")
        
        # Behavior evaluation form
        st.markdown("---")
        with st.form("behavior_evaluation", clear_on_submit=False):
            st.subheader(f"Evaluation for: {selected_student} ({reg_number})")
            
            st.markdown("### Rate each behavior category:")
            
            # Behavior categories
            behavior_categories = {
                'punctuality': 'Punctuality',
                'attendance': 'Attendance',
                'manners': 'Manners',
                'general_behavior': 'General Behavior',
                'organisational_skills': 'Organisational Skills',
                'adherence_to_uniform': 'Adherence to Uniform',
                'leadership_skills': 'Leadership Skills',
                'commitment_to_school': 'Commitment to School',
                'cooperation_with_peers': 'Cooperation with Peers',
                'cooperation_with_staff': 'Cooperation with Staff',
                'participation_in_lessons': 'Participation in Lessons',
                'completion_of_homework': 'Completion of Homework'
            }
            
            ratings = ['Excellent', 'Good', 'Satisfactory', 'Cause of Concern']
            
            # Create rating selections
            behavior_ratings = {}
            
            # Split into 2 columns for better layout
            col1, col2 = st.columns(2)
            
            items = list(behavior_categories.items())
            half = len(items) // 2
            
            with col1:
                for field, label in items[:half]:
                    default_value = 0
                    if existing_behavior:
                        try:
                            existing_val = getattr(existing_behavior, field)
                            if existing_val in ratings:
                                default_value = ratings.index(existing_val)
                        except (ValueError, AttributeError):
                            default_value = 0
                    
                    behavior_ratings[field] = st.selectbox(
                        f"**{label}**",
                        ratings,
                        index=default_value,
                        key=f"behavior_{field}"
                    )
            
            with col2:
                for field, label in items[half:]:
                    default_value = 0
                    if existing_behavior:
                        try:
                            existing_val = getattr(existing_behavior, field)
                            if existing_val in ratings:
                                default_value = ratings.index(existing_val)
                        except (ValueError, AttributeError):
                            default_value = 0
                    
                    behavior_ratings[field] = st.selectbox(
                        f"**{label}**",
                        ratings,
                        index=default_value,
                        key=f"behavior_{field}"
                    )
            
            st.markdown("---")
            submit_button = st.form_submit_button("💾 Save Behavior Evaluation", type="primary")
            
            if submit_button:
                # Show what we're about to save
                st.write("**Debug: Attempting to save...**")
                st.write(f"Student ID: {student_id}")
                st.write(f"Term ID: {active_term.id}")
                st.write(f"User ID: {st.session_state.user_id}")
                st.write(f"Ratings collected: {len(behavior_ratings)} items")
                
                try:
                    if existing_behavior:
                        st.write("**Debug: Updating existing record...**")
                        # Update existing behavior evaluation
                        for field, rating in behavior_ratings.items():
                            setattr(existing_behavior, field, rating)
                            st.write(f"Set {field} = {rating}")
                        
                        existing_behavior.evaluated_by = st.session_state.user_id
                        existing_behavior.evaluated_at = datetime.now().isoformat()
                        
                        session.commit()
                        session.flush()
                        
                        log_audit(session, st.session_state.user_id, "update_behavior", 
                                 f"{selected_student} - {active_term.term_name}")
                        
                        st.success(f"✅ Behavior evaluation UPDATED for {selected_student}!")
                        st.balloons()
                        
                    else:
                        st.write("**Debug: Creating new record...**")
                        # Create new behavior evaluation
                        new_behavior = ClassroomBehavior(
                            student_id=student_id,
                            term_id=active_term.id,
                            evaluated_by=st.session_state.user_id,
                            evaluated_at=datetime.now().isoformat()
                        )
                        
                        # Set each behavior rating
                        for field, rating in behavior_ratings.items():
                            setattr(new_behavior, field, rating)
                            st.write(f"Set {field} = {rating}")
                        
                        session.add(new_behavior)
                        session.commit()
                        session.flush()
                        
                        # Verify it was saved
                        verify = session.query(ClassroomBehavior).filter_by(
                            student_id=student_id,
                            term_id=active_term.id
                        ).first()
                        
                        if verify:
                            st.success(f"✅ Behavior evaluation CREATED for {selected_student}! (ID: {verify.id})")
                        else:
                            st.warning("⚠️ Record created but verification failed")
                        
                        log_audit(session, st.session_state.user_id, "submit_behavior", 
                                 f"{selected_student} - {active_term.term_name}")
                        st.balloons()
                    
                    # Force rerun after successful save
                    import time
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    session.rollback()
                    st.error(f"❌ ERROR saving behavior evaluation!")
                    st.error(f"Error message: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    
                    # Show full traceback
                    import traceback
                    st.code(traceback.format_exc())
                    
                    st.error(f"Debug info:")
                    st.error(f"- Student ID: {student_id}")
                    st.error(f"- Term ID: {active_term.id}")
                    st.error(f"- User ID: {st.session_state.user_id}")
        
        # View existing evaluations
        st.markdown("---")
        st.subheader("Existing Behavior Evaluations for This Class")
        
        # Get all behavior evaluations for this class
        behavior_df = pd.read_sql(f"""
            SELECT cb.*, s.name, s.registration_number
            FROM classroom_behavior cb
            JOIN students s ON cb.student_id = s.id
            WHERE cb.term_id = {active_term.id}
            AND s.class_name = '{class_name}'
            ORDER BY s.name
        """, ENGINE)
        
        if not behavior_df.empty:
            st.success(f"✅ Found {len(behavior_df)} evaluations")
            
            # Summary table
            summary_data = []
            for _, row in behavior_df.iterrows():
                summary_data.append({
                    'Student': row['name'],
                    'Registration': row['registration_number'],
                    'Punctuality': row['punctuality'],
                    'Attendance': row['attendance'],
                    'General Behavior': row['general_behavior'],
                    'Date': row['evaluated_at'][:10] if row['evaluated_at'] else 'N/A'
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
        else:
            st.info("📝 No behavior evaluations completed yet for this class")
    
    # Admin view (simplified)
    elif st.session_state.user_role == 'admin':
        st.info("📋 As an administrator, you can view all behavior evaluations.")
        
        classes = pd.read_sql("SELECT DISTINCT class_name FROM students ORDER BY class_name", ENGINE)
        
        if classes.empty:
            st.info("No classes available")
            session.close()
            st.stop()
        
        selected_class = st.selectbox("Select Class to View", classes['class_name'].tolist())
        
        behavior_df = pd.read_sql(f"""
            SELECT cb.*, s.name, s.registration_number, u.name as evaluator_name
            FROM classroom_behavior cb
            JOIN students s ON cb.student_id = s.id
            JOIN users u ON cb.evaluated_by = u.id
            WHERE cb.term_id = {active_term.id}
            AND s.class_name = '{selected_class}'
            ORDER BY s.name
        """, ENGINE)
        
        if not behavior_df.empty:
            st.success(f"✅ Found {len(behavior_df)} evaluations for {selected_class}")
            st.dataframe(behavior_df[['name', 'registration_number', 'evaluator_name', 
                                         'punctuality', 'attendance', 'general_behavior']], 
                           use_container_width=True)
        else:
            st.info(f"No evaluations yet for {selected_class}")
                
            class_teacher = session.query(User).filter_by(class_teacher_for=selected_class).first()
            if class_teacher:
                st.info(f"📌 Class Teacher: {class_teacher.name} ({class_teacher.email})")
            else:
                st.warning(f"No class teacher assigned to {selected_class}")
    
    session.close()

elif page == "Discipline Reports":
    st.header("⚠️ Discipline Reports")
    session = Session()
    
    if st.session_state.user_role == 'teacher':
        st.subheader("Submit Discipline Report")
        
        students = pd.read_sql("SELECT id, name, class_name FROM students ORDER BY name", ENGINE)
        
        if not students.empty:
            with st.form("discipline_report"):
                student_name = st.selectbox("Select Student", students['name'].tolist())
                incident_date = st.date_input("Incident Date")
                incident_type = st.selectbox("Incident Type", 
                                            ["Behavioral", "Academic", "Attendance", "Other"])
                description = st.text_area("Description of Incident*", height=150)
                action_taken = st.text_area("Action Taken by Teacher", height=100)
                
                submit_button = st.form_submit_button("Submit Report")
                
                if submit_button:
                    if not description:
                        st.error("Description is required")
                    else:
                        student_id = students[students['name'] == student_name].iloc[0]['id']
                        
                        report = DisciplineReport(
                            student_id=student_id,
                            reported_by=st.session_state.user_id,
                            incident_date=str(incident_date),
                            incident_type=incident_type,
                            description=description,
                            action_taken=action_taken,
                            status="Pending"
                        )
                        session.add(report)
                        session.commit()
                        log_audit(session, st.session_state.user_id, "submit_discipline_report", student_name)
                        st.success("✅ Discipline report submitted to admin")
                        st.rerun()
        else:
            st.info("No students available")
        
        st.markdown("---")
        st.subheader("My Submitted Reports")
        my_reports = pd.read_sql(f"""
            SELECT dr.*, s.name as student_name, s.class_name
            FROM discipline_reports dr
            JOIN students s ON dr.student_id = s.id
            WHERE dr.reported_by = {st.session_state.user_id}
            ORDER BY dr.created_at DESC
        """, ENGINE)
        
        if not my_reports.empty:
            for _, report in my_reports.iterrows():
                with st.expander(f"📋 {report['student_name']} - {report['incident_type']} - {report['status']}"):
                    st.write(f"**Date:** {report['incident_date']}")
                    st.write(f"**Class:** {report['class_name']}")
                    st.write(f"**Description:** {report['description']}")
                    st.write(f"**Action Taken:** {report['action_taken']}")
                    if report['admin_notes']:
                        st.info(f"**Admin Notes:** {report['admin_notes']}")
        else:
            st.info("No reports submitted yet")
    
    elif st.session_state.user_role == 'admin':
        st.subheader("All Discipline Reports")
        
        status_filter = st.selectbox("Filter by Status", ["All", "Pending", "Reviewed", "Resolved"])
        
        if status_filter == "All":
            reports = pd.read_sql("""
                SELECT dr.*, s.name as student_name, s.class_name, u.name as teacher_name
                FROM discipline_reports dr
                JOIN students s ON dr.student_id = s.id
                JOIN users u ON dr.reported_by = u.id
                ORDER BY dr.created_at DESC
            """, ENGINE)
        else:
            reports = pd.read_sql(f"""
                SELECT dr.*, s.name as student_name, s.class_name, u.name as teacher_name
                FROM discipline_reports dr
                JOIN students s ON dr.student_id = s.id
                JOIN users u ON dr.reported_by = u.id
                WHERE dr.status = '{status_filter}'
                ORDER BY dr.created_at DESC
            """, ENGINE)
        
        if not reports.empty:
            for _, report in reports.iterrows():
                status_color = {
                    "Pending": "🔴",
                    "Reviewed": "🟡",
                    "Resolved": "🟢"
                }
                
                with st.expander(f"{status_color.get(report['status'], '⚪')} {report['student_name']} ({report['class_name']}) - {report['incident_type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Reported by:** {report['teacher_name']}")
                        st.write(f"**Date of Incident:** {report['incident_date']}")
                        st.write(f"**Submitted on:** {report['created_at']}")
                    with col2:
                        st.write(f"**Type:** {report['incident_type']}")
                        st.write(f"**Status:** {report['status']}")
                    
                    st.markdown("**Description:**")
                    st.write(report['description'])
                    
                    if report['action_taken']:
                        st.markdown("**Teacher's Action:**")
                        st.write(report['action_taken'])
                    
                    if report['admin_notes']:
                        st.markdown("**Admin Notes:**")
                        st.info(report['admin_notes'])
                    
                    with st.form(f"admin_action_{report['id']}"):
                        new_status = st.selectbox("Update Status", 
                                                 ["Pending", "Reviewed", "Resolved"],
                                                 index=["Pending", "Reviewed", "Resolved"].index(report['status']))
                        admin_notes = st.text_area("Admin Notes", value=report['admin_notes'] or "")
                        
                        if st.form_submit_button("Update Report"):
                            rep = session.query(DisciplineReport).get(report['id'])
                            rep.status = new_status
                            rep.admin_notes = admin_notes
                            session.commit()
                            log_audit(session, st.session_state.user_id, "update_discipline_report", 
                                     f"{report['student_name']} - {new_status}")
                            st.success("✅ Report updated")
                            st.rerun()
        else:
            st.info("No discipline reports found")
    
    session.close()

elif page == "Generate Reports" and st.session_state.user_role == 'admin':
    st.header("📄 Generate Student Reports")
    session = Session()
    
    students = pd.read_sql("SELECT * FROM students ORDER BY class_name, name", ENGINE)
    terms = pd.read_sql("SELECT * FROM academic_terms ORDER BY year DESC, term_number DESC", ENGINE)
    
    if students.empty or terms.empty:
        st.warning("Need students and terms to generate reports")
        session.close()
        st.stop()
    
    design = session.query(ReportDesign).first()
    
    st.subheader("Select Report Generation Options")
    
    report_mode = st.radio("Generate reports for:", ["Individual Student", "Whole Class"], horizontal=True)
    
    selected_term = st.selectbox("Select Term", terms['term_name'].tolist())
    term_data = terms[terms['term_name'] == selected_term].iloc[0]
    
    if report_mode == "Individual Student":
        selected_student = st.selectbox("Select Student", students['name'].tolist())
        student_data = students[students['name'] == selected_student].iloc[0]
        
        # FIXED QUERY - properly get marks
        marks_query = f"""
            SELECT m.*, u.name as teacher_name 
            FROM marks m
            JOIN users u ON m.submitted_by = u.id
            WHERE m.student_id = {student_data['id']} AND m.term_id = {term_data['id']}
            ORDER BY m.subject
        """
        marks = pd.read_sql(marks_query, ENGINE)
        
        # Get behavior data - FIXED VERSION
        behavior_query = f"""
            SELECT punctuality, attendance, manners, general_behavior, 
                   organisational_skills, adherence_to_uniform, leadership_skills,
                   commitment_to_school, cooperation_with_peers, cooperation_with_staff,
                   participation_in_lessons, completion_of_homework
            FROM classroom_behavior
            WHERE student_id = {student_data['id']} AND term_id = {term_data['id']}
        """
        behavior_result = pd.read_sql(behavior_query, ENGINE)
        
        behavior_data = None
        if not behavior_result.empty:
            # Convert DataFrame row to dictionary
            behavior_data = behavior_result.iloc[0].to_dict()
        
        # Get decision data (only for term 3)
        decision_data = None
        if term_data['term_number'] == 3:
            decision_query = f"""
                SELECT decision, notes
                FROM student_decisions
                WHERE student_id = {student_data['id']} AND term_id = {term_data['id']}
            """
            decision_result = pd.read_sql(decision_query, ENGINE)
            if not decision_result.empty:
                decision_data = decision_result.iloc[0].to_dict()
        
        if not marks.empty:
            st.subheader(f"Results Preview for {selected_student}")
            
            display_df = marks[[
                'subject', 'coursework_out_of_20', 'midterm_out_of_20', 
                'endterm_out_of_60', 'total', 'grade', 'comment', 'teacher_name'
            ]].copy()
            
            display_df.columns = ['Subject', 'CW/20', 'MOT/20', 'EOT/60', 'Total', 'Grade', 'Comment', 'Teacher']
            st.dataframe(display_df, use_container_width=True)
            
            overall_avg = marks['total'].mean()
            overall_grade = get_grade(overall_avg)
            
            # Parent-friendly average interpretation
            if overall_grade == "A*":
                avg_comment = "Outstanding Performance"
            elif overall_grade == "A":
                avg_comment = "Excellent Performance"
            elif overall_grade == "B":
                avg_comment = "Good Performance"
            elif overall_grade == "C":
                avg_comment = "Satisfactory Performance"
            elif overall_grade == "D":
                avg_comment = "Needs Improvement"
            elif overall_grade == "E":
                avg_comment = "Poor Performance"
            else:
                avg_comment = "Unsatisfactory Performance"
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Overall Average", f"{overall_avg:.0f}/100")
            with col4:
                st.metric("Overall Grade", f"{overall_grade} - {avg_comment}")
            
            # Show decision if term 3
            if term_data['term_number'] == 3 and decision_data:
                st.metric("Decision", decision_data.get('decision', 'Pending'))
            
            if st.button("📄 Generate PDF Report", use_container_width=True):
                try:
                    pdf_data = generate_pdf_report(student_data, term_data, marks, design, behavior_data, decision_data)
                    
                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf_data,
                        f"{selected_student}_{selected_term}_report.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                    log_audit(session, st.session_state.user_id, "generate_report", 
                             f"Individual: {selected_student} - {selected_term}")
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
        else:
            st.info(f"❌ No marks found for {selected_student} in {selected_term}")
            st.info(f"Debug: Student ID = {student_data['id']}, Term ID = {term_data['id']}")
            
            # Show if student has ANY marks
            any_marks = pd.read_sql(f"SELECT COUNT(*) as count FROM marks WHERE student_id = {student_data['id']}", ENGINE)
            st.info(f"This student has {any_marks['count'].iloc[0]} marks in total across all terms")
    
    else:  # Whole Class
        classes = pd.read_sql("SELECT DISTINCT class_name FROM students ORDER BY class_name", ENGINE)
        selected_class = st.selectbox("Select Class", classes['class_name'].tolist())
        
        class_students = students[students['class_name'] == selected_class]
        
        st.info(f"📊 Found {len(class_students)} students in {selected_class}")
        
        students_with_marks = []
        students_without_marks = []
        
        for _, student in class_students.iterrows():
            marks = pd.read_sql(f"""
                SELECT m.*, u.name as teacher_name 
                FROM marks m
                JOIN users u ON m.submitted_by = u.id
                WHERE m.student_id = {student['id']} AND m.term_id = {term_data['id']}
            """, ENGINE)
            
            if not marks.empty:
                students_with_marks.append(student['name'])
            else:
                students_without_marks.append(student['name'])
        
        col5, col6 = st.columns(2)
        with col5:
            st.success(f"✅ {len(students_with_marks)} students with marks")
        with col6:
            if students_without_marks:
                st.warning(f"⚠️ {len(students_without_marks)} students without marks")
        
        if students_without_marks:
            with st.expander("⚠️ Students without marks"):
                st.write(", ".join(students_without_marks))
        
        if students_with_marks:
            if st.button(f"📄 Generate All Reports for {selected_class}", use_container_width=True):
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for student_name in students_with_marks:
                        student_data = class_students[class_students['name'] == student_name].iloc[0]
                        
                        marks = pd.read_sql(f"""
                            SELECT m.*, u.name as teacher_name 
                            FROM marks m
                            JOIN users u ON m.submitted_by = u.id
                            WHERE m.student_id = {student_data['id']} AND m.term_id = {term_data['id']}
                        """, ENGINE)
                        
                        # Get behavior data - FIXED VERSION
                        behavior_query = f"""
                            SELECT punctuality, attendance, manners, general_behavior, 
                                   organisational_skills, adherence_to_uniform, leadership_skills,
                                   commitment_to_school, cooperation_with_peers, cooperation_with_staff,
                                   participation_in_lessons, completion_of_homework
                            FROM classroom_behavior
                            WHERE student_id = {student_data['id']} AND term_id = {term_data['id']}
                        """
                        behavior_result = pd.read_sql(behavior_query, ENGINE)
                        
                        behavior_data = None
                        if not behavior_result.empty:
                            # Convert DataFrame row to dictionary
                            behavior_data = behavior_result.iloc[0].to_dict()
                        
                        # Get decision data (only for term 3)
                        decision_data = None
                        if term_data['term_number'] == 3:
                            decision_query = f"""
                                SELECT decision, notes
                                FROM student_decisions
                                WHERE student_id = {student_data['id']} AND term_id = {term_data['id']}
                            """
                            decision_result = pd.read_sql(decision_query, ENGINE)
                            if not decision_result.empty:
                                decision_data = decision_result.iloc[0].to_dict()
                        
                        if not marks.empty:
                            pdf_data = generate_pdf_report(student_data, term_data, marks, design, behavior_data, decision_data)
                            filename = f"{student_name}_{selected_term}_report.pdf"
                            zip_file.writestr(filename, pdf_data)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    f"⬇️ Download All {len(students_with_marks)} Reports (ZIP)",
                    zip_buffer.getvalue(),
                    f"{selected_class}_{selected_term}_reports.zip",
                    "application/zip",
                    use_container_width=True
                )
                
                log_audit(session, st.session_state.user_id, "generate_reports", 
                         f"Bulk: {selected_class} - {selected_term} - {len(students_with_marks)} reports")
                st.success(f"✅ Generated {len(students_with_marks)} reports!")
        else:
            st.warning("No students in this class have marks for the selected term")
    
    session.close()

elif page == "Report Design" and st.session_state.user_role == 'admin':
    st.header("🎨 Report Design Customization")
    
    st.info("💡 Customize how your school reports look. All changes will be reflected in newly generated reports.")
    
    tab1, tab2, tab3 = st.tabs(["School Information", "Logo & Colors", "Preview"])
    
    # Create session for each tab to avoid stale sessions
    session = Session()
    design = session.query(ReportDesign).first()
    
    with tab1:
        st.subheader("School Details")
        
        with st.form("school_info"):
            school_name = st.text_input("School Name*", value=design.school_name)
            school_subtitle = st.text_input("School Subtitle", value=design.school_subtitle or "")
            school_address = st.text_input("School Address", value=design.school_address or "")
            school_po_box = st.text_input("P.O. Box", value=design.school_po_box or "")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                school_phone = st.text_input("Phone Number", value=design.school_phone or "")
            with col2:
                school_email = st.text_input("Email", value=design.school_email or "")
            with col3:
                school_website = st.text_input("Website", value=design.school_website or "")
            
            report_footer = st.text_area("Report Footer (optional)", 
                                        value=design.report_footer or "",
                                        help="Add any additional text to appear at bottom of reports")
            
            if st.form_submit_button("💾 Save School Information", use_container_width=True):
                try:
                    # Get fresh session and design object
                    save_session = Session()
                    save_design = save_session.query(ReportDesign).first()
                    
                    save_design.school_name = school_name
                    save_design.school_subtitle = school_subtitle
                    save_design.school_address = school_address
                    save_design.school_po_box = school_po_box
                    save_design.school_phone = school_phone
                    save_design.school_email = school_email
                    save_design.school_website = school_website
                    save_design.report_footer = report_footer
                    
                    save_session.commit()
                    log_audit(save_session, st.session_state.user_id, "update_report_design", "School information")
                    save_session.close()
                    
                    st.success("✅ School information updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving: {str(e)}")
                    if 'save_session' in locals():
                        save_session.rollback()
                        save_session.close()
    
    with tab2:
        st.subheader("Logo & Visual Design")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Upload School Logo**")
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                try:
                    # Convert to base64
                    bytes_data = uploaded_file.getvalue()
                    b64 = base64.b64encode(bytes_data).decode()
                    
                    # Save to database
                    design.logo_data = b64
                    session.commit()
                    st.success("✅ Logo uploaded successfully!")
                    st.image(uploaded_file, width=300)
                except Exception as e:
                    st.error(f"❌ Error uploading logo: {str(e)}")
            elif design.logo_data:
                try:
                    if design.logo_data.startswith('http'):
                        # It's a URL
                        st.image(design.logo_data, width=300)
                    else:
                        # It's base64
                        logo_bytes = base64.b64decode(design.logo_data)
                        st.image(logo_bytes, width=300)
                except:
                    st.error("Error displaying existing logo")
            
            # NEW: Quick insert Empower logo
            st.markdown("---")
            st.markdown("**🚀 Quick Insert Empower Logo**")
            if st.button("Use Empower Academy Logo", use_container_width=True):
                empower_logo_url = "https://z-cdn-media.chatglm.cn/files/a7ca3e7c-8f26-410d-94e5-84b20d17eaed_empower-logo.png?auth_key=1863023354-290424df56d14d3b9f2ee211186220cf-0-e728679b39cedb32228a3c796ca046cf"
                logo_b64 = download_logo_from_url(empower_logo_url)
                if logo_b64:
                    design.logo_data = logo_b64
                    session.commit()
                    st.success("✅ Empower Academy logo added successfully!")
                    st.rerun()
        
        with col2:
            st.markdown("**Or Enter Logo URL**")
            logo_url = st.text_input("Enter logo URL:", 
                                     value=design.logo_data if design.logo_data and design.logo_data.startswith('http') else "",
                                     help="Enter direct URL to your logo image")
            
            if st.button("Load Logo from URL", use_container_width=True):
                if logo_url:
                    logo_b64 = download_logo_from_url(logo_url)
                    if logo_b64:
                        design.logo_data = logo_url  # Store URL directly
                        session.commit()
                        st.success("✅ Logo URL saved successfully!")
                        st.rerun()
            
            st.markdown("**Primary Color**")
            st.write("Choose main color for headers and tables")
            
            primary_color = st.color_picker("Primary Color", value=design.primary_color)
            
            if st.button("💾 Save Color", use_container_width=True):
                design.primary_color = primary_color
                session.commit()
                log_audit(session, st.session_state.user_id, "update_report_design", f"Color: {primary_color}")
                st.success("✅ Color updated!")
                st.rerun()
            
            st.markdown("**Preview:**")
            st.markdown(f"<div style='background-color: {design.primary_color}; color: white; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold;'>Sample Header Text</div>", unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Report Preview")
        st.write("This is how your report header will look:")
        
        st.markdown("---")
        
        if design.logo_data:
            try:
                if design.logo_data.startswith('http'):
                    # It's a URL
                    st.image(design.logo_data, width=150)
                else:
                    # It's base64
                    logo_bytes = base64.b64decode(design.logo_data)
                    st.image(logo_bytes, width=150)
            except:
                st.error("Error loading logo")
        
        st.markdown(f"<h2 style='text-align: center; color: {design.primary_color};'>{design.school_name}</h2>", unsafe_allow_html=True)
        
        if design.school_subtitle:
            st.markdown(f"<p style='text-align: center;'>{design.school_subtitle}</p>", unsafe_allow_html=True)
        if design.school_address:
            st.markdown(f"<p style='text-align: center;'>{design.school_address}</p>", unsafe_allow_html=True)
        if design.school_po_box:
            st.markdown(f"<p style='text-align: center;'>{design.school_po_box}</p>", unsafe_allow_html=True)
        
        contact_parts = []
        if design.school_phone:
            contact_parts.append(f"Tel: {design.school_phone}")
        if design.school_email:
            contact_parts.append(f"Email: {design.school_email}")
        if design.school_website:
            contact_parts.append(f"Web: {design.school_website}")
        
        if contact_parts:
            st.markdown(f"<p style='text-align: center;'>{' | '.join(contact_parts)}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        sample_data = {
            'Subject': ['Mathematics', 'English', 'Physics'],
            'CW': [18, 17, 19],
            'MOT': [16, 18, 17],
            'EOT': [55, 52, 54],
            'Total': [89, 87, 90],
            'Grade': ['A', 'A', 'A*']
        }
        st.dataframe(sample_data, use_container_width=True)
        
        if design.report_footer:
            st.markdown("---")
            st.markdown(f"<p style='text-align: center; font-style: italic;'>{design.report_footer}</p>", unsafe_allow_html=True)
    
    session.close()

elif page == "Data Export" and st.session_state.user_role == 'admin':
    st.header("📥 Data Export")
    session = Session()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Students")
        if st.button("Download Students Data", use_container_width=True):
            # Create backup before export
            auto_backup_before_critical_operation("data_export_students")
            
            df = pd.read_sql("SELECT * FROM students", ENGINE)
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv, "students.csv", "text/csv", use_container_width=True)
            else:
                st.info("No data to export")
    
    with col2:
        st.subheader("Export Staff")
        if st.button("Download Staff Data", use_container_width=True):
            df = pd.read_sql("SELECT * FROM users WHERE role = 'teacher'", ENGINE)
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv, "staff.csv", "text/csv", use_container_width=True)
            else:
                st.info("No data to export")
    
    st.markdown("---")
    st.subheader("Export Results")
    
    terms = pd.read_sql("SELECT * FROM academic_terms ORDER BY year DESC, term_number DESC", ENGINE)
    if not terms.empty:
        selected_term = st.selectbox("Select Term", terms['term_name'].tolist())
        term_id = terms[terms['term_name'] == selected_term].iloc[0]['id']
        
        if st.button("Download Results for Selected Term", use_container_width=True):
            # Create backup before export
            auto_backup_before_critical_operation("data_export_results")
            
            marks_df = pd.read_sql(f"""
                SELECT 
                    s.name as student_name,
                    s.class_name,
                    s.year,
                    m.subject,
                    m.coursework_out_of_20,
                    m.midterm_out_of_20,
                    m.endterm_out_of_60,
                    m.total,
                    m.grade,
                    m.comment,
                    u.name as teacher_name,
                    at.term_name
                FROM marks m
                JOIN students s ON m.student_id = s.id
                JOIN users u ON m.submitted_by = u.id
                JOIN academic_terms at ON m.term_id = at.id
                WHERE m.term_id = {term_id}
                ORDER BY s.class_name, s.name, m.subject
            """, ENGINE)
            
            if not marks_df.empty:
                csv = marks_df.to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv, 
                                  f"results_{selected_term.replace(' ', '_')}.csv", "text/csv",
                                  use_container_width=True)
            else:
                st.info("No results for this term")
    else:
        st.info("No terms available")
    
    st.markdown("---")
    st.subheader("Export Discipline Reports")
    if st.button("Download All Discipline Reports", use_container_width=True):
        reports_df = pd.read_sql("""
            SELECT 
                s.name as student_name,
                s.class_name,
                u.name as reported_by,
                dr.incident_date,
                dr.incident_type,
                dr.description,
                dr.action_taken,
                dr.status,
                dr.admin_notes,
                dr.created_at
            FROM discipline_reports dr
            JOIN students s ON dr.student_id = s.id
            JOIN users u ON dr.reported_by = u.id
            ORDER BY dr.created_at DESC
        """, ENGINE)
        
        if not reports_df.empty:
            csv = reports_df.to_csv(index=False)
            st.download_button("⬇️ Download Reports CSV", csv, "discipline_reports.csv", "text/csv",
                             use_container_width=True)
        else:
            st.info("No discipline reports")
    
    session.close()

elif page == "Change Login Details":
    st.header("🔐 Change Login Details")
    session = Session()
    user = session.query(User).get(st.session_state.user_id)
    
    with st.form("change_login"):
        st.subheader("Update Your Credentials")
        new_email = st.text_input("New Email/Username", value=user.email)
        current_pass = st.text_input("Current Password*", type="password")
        new_pass = st.text_input("New Password (leave blank to keep current)", type="password")
        confirm_pass = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Login Details", use_container_width=True):
            if hashlib.sha256(current_pass.encode()).hexdigest() != user.password_hash:
                st.error("❌ Current password is incorrect")
            elif new_pass and new_pass != confirm_pass:
                st.error("❌ New passwords don't match")
            elif session.query(User).filter(User.email == new_email, User.id != user.id).first():
                st.error("❌ Email already taken by another user")
            else:
                user.email = new_email
                if new_pass:
                    user.password_hash = hashlib.sha256(new_pass.encode()).hexdigest()
                session.commit()
                log_audit(session, st.session_state.user_id, "change_login", new_email)
                st.success("✅ Login details updated! Please login again with new credentials.")
                st.session_state.logged_in = False
                session.close()
                st.rerun()
    
    session.close()

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("💡 **Empower Reports v4.0**\n\nFeatures:\n- ✅ Results submission\n- ✅ Logo upload\n- ✅ Report generation\n- ✅ PDF formatting\n- ✅ Classroom behavior\n- ✅ Parent-friendly format\n- ✅ One-page layout\n- ✅ Performance analytics\n- ✅ Improvement tracking\n- ✅ Presentation mode\n- ✅ Student decisions\n- ✅ Visitation Day management\n- ✅ Local storage\n- ✅ Backup & restore\n\n[View Documentation](https://github.com)")

# Backup reminder for admins
if st.session_state.user_role == 'admin':
    session = Session()
    total_students = session.query(Student).count()
    total_marks = session.query(Mark).count()
    session.close()
    
    if total_students > 0 or total_marks > 0:
        st.sidebar.warning("📌 **Reminder**: Export your data regularly!\n\nGo to **Data Export** to backup.")

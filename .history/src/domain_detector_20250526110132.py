import pandas as pd
import numpy as np
from typing import Dict, List, Any
import re

class EducationDomainDetector:
    def __init__(self):
        self.education_keywords = {
            'student_identifiers': [
                'student', 'pupil', 'learner', 'id', 'roll', 'enrollment',
                'matriculation', 'admission'
            ],
            'academic_performance': [
                'grade', 'score', 'mark', 'result', 'gpa', 'cgpa', 'percentage',
                'exam', 'test', 'quiz', 'assignment', 'final', 'midterm',
                'performance', 'achievement', 'rank', 'position'
            ],
            'subjects_courses': [
                'math', 'science', 'english', 'history', 'geography', 'physics',
                'chemistry', 'biology', 'literature', 'course', 'subject',
                'module', 'unit', 'credit', 'semester', 'term'
            ],
            'demographics': [
                'age', 'gender', 'class', 'year', 'level', 'section',
                'batch', 'cohort', 'group'
            ],
            'attendance_behavior': [
                'attendance', 'absent', 'present', 'tardy', 'punctuality',
                'behavior', 'discipline', 'conduct'
            ],
            'academic_support': [
                'tutor', 'counselor', 'mentor', 'support', 'intervention',
                'remedial', 'enrichment', 'special', 'needs'
            ],
            'family_background': [
                'parent', 'guardian', 'family', 'income', 'education',
                'occupation', 'socioeconomic', 'background'
            ],
            'school_environment': [
                'school', 'university', 'college', 'institution', 'campus',
                'classroom', 'teacher', 'instructor', 'faculty'
            ]
        }
    
    def detect_education_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if the dataset belongs to education domain"""
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.education_keywords.keys()}
        total_matches = 0
        
        # Check column names against education keywords
        for category, keywords in self.education_keywords.items():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                if matching_columns:
                    detected_features[category].extend(matching_columns)
                    total_matches += len(matching_columns)
        
        # Additional checks for data patterns
        pattern_score = self._check_data_patterns(df)
        
        # Calculate confidence score
        total_keywords = sum(len(keywords) for keywords in self.education_keywords.values())
        keyword_confidence = min(total_matches / 10, 1.0)  # Normalize to 0-1
        pattern_confidence = pattern_score / 10  # Normalize to 0-1
        
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)
        
        # Determine if it's education domain
        is_education = overall_confidence >= 0.3
        
        return {
            'is_education': is_education,
            'confidence': overall_confidence,
            'detected_features': detected_features,
            'total_matches': total_matches,
            'pattern_score': pattern_score
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for typical education data patterns"""
        score = 0
        
        # Check for grade-like numeric columns (0-100 range)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].max() <= 100 and df[col].min() >= 0:
                if any(keyword in col.lower() for keyword in ['grade', 'score', 'mark', 'result']):
                    score += 2
                else:
                    score += 1
        
        # Check for age columns (typical student age range)
        for col in numeric_cols:
            if 'age' in col.lower() and df[col].max() <= 30 and df[col].min() >= 5:
                score += 2
        
        # Check for categorical data typical in education
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            if len(unique_values) < 20:  # Reasonable number of categories
                col_lower = col.lower()
                values_lower = [str(v).lower() for v in unique_values]
                
                # Check for common education categories
                if any(keyword in col_lower for keyword in ['gender', 'class', 'level', 'grade']):
                    score += 1
                
                if any(val in values_lower for val in ['male', 'female', 'pass', 'fail', 'a', 'b', 'c', 'd', 'f']):
                    score += 1
        
        return min(score, 10)  # Cap at 10
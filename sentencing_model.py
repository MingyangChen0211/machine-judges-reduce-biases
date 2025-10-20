#!/usr/bin/env python3
"""
Theft Crime Sentencing Prediction System
Academic Research Code for Legal Feature Extraction and ML Modeling
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import math
import re
from datetime import datetime
import warnings
from typing import Dict, List, Tuple
import joblib

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


class LegalFeatureExtractor:
    """Extract core legal features from court documents using one-hot encoding"""
    
    def __init__(self):
        # Define mutually exclusive feature groups
        self.strict_mutex_groups = [
            ['attempt_prep', 'attempt_incomp', 'attempt_stop'],  # Crime attempt stages
            ['theft_pickpocket', 'theft_household'],  # Theft methods
            ['age_minor', 'age_elderly'],  # Age categories
        ]
        
        # Core legal sentencing features (28 binary features)
        self.core_features = {
            # Crime attempt stages
            'attempt_prep': ['犯罪预备', '预备犯', '盗窃预备'],
            'attempt_incomp': ['犯罪未遂', '未遂犯', '盗窃未遂'],
            'attempt_stop': ['犯罪中止', '中止犯', '盗窃中止'],
            
            # Amount levels
            'amount_large': ['数额较大', '价值.*较大', '较大数额'],
            'amount_huge': ['数额巨大', '价值.*巨大', '巨大数额'],
            'amount_extraordinary': ['数额特别巨大', '特别巨大数额'],
            
            # Theft methods
            'weapon_carry': ['携带凶器', '持凶器'],
            'theft_multiple': ['多次盗窃', '多次犯'],
            'theft_roaming': ['流窜作案', '流窜犯罪'],
            'theft_pickpocket': ['扒窃'],
            'theft_household': ['入户盗窃', '入室盗窃', '进入.*室.*盗', '在.*室内.*盗', '入.*房.*盗', 
                               '进入.*家中', '进入.*房间', '入户', '入室'],
            
            # Mitigating circumstances
            'surrender': ['系自首', '属自首', '构成自首'],
            'meritorious': ['立功表现', '有立功'],
            'confession_formal': ['系坦白', '属坦白', '如实供述.*坦白', '坦白.*供述'],
            'confession_truthful': ['如实供述', '如实交代', '主动供述'],
            'plead_voluntary': ['自愿认罪', '主动认罪'],
            'plead_accept_punishment': ['认罪认罚', '签署.*认罪认罚.*具结书', '适用认罪认罚'],
            
            # Aggravating circumstances
            'recidivist': ['累犯', '构成累犯'],
            'prior_record': ['前科', '有犯罪前科'],
            
            # Special subjects
            'age_minor': ['未成年', '未满18周岁'],
            'age_elderly': ['老年人', '75周岁'],
            'disability': ['残疾'],
            'mental_illness': ['精神病', '精神疾病'],
            
            # Victim-related
            'victim_forgiveness': ['谅解'],
            'settlement': ['和解'],
            'compensation': ['赔偿', '退赔'],
            
            # Special circumstances
            'gang_crime': ['黑恶势力'],
            'disaster_period': ['灾害'],
        }
    
    def preprocess_text(self, text: str) -> str:
        """Remove legal article citations to avoid false positives"""
        if not isinstance(text, str):
            return ""
        
        # Remove appendix sections
        if '附录' in text:
            text = text.split('附录')[0]
        
        # Remove law article citations
        law_patterns = [
            r'第二百六十四条.*?处罚金或者没收财产[。\.]',
            r'第六十七条.*?可以减轻处罚[。\.]',
            r'第\d+条[^。\.]*?盗窃[^。\.]*?处三年以下有期徒刑.*?',
            r'附[：:，,\s]*法律条文[：:，,\s]*.*',
            r'依照《.*?》第\d+条.*?规定[：:，,；;。\.\s]*.*?',
        ]
        
        for pattern in law_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def detect_conflicts(self, features_dict: Dict[str, int]) -> List[str]:
        """Detect mutually exclusive feature conflicts"""
        conflicts = []
        for group in self.strict_mutex_groups:
            active = [f for f in group if features_dict.get(f, 0) == 1]
            if len(active) > 1:
                conflicts.append(f"Mutex conflict: {', '.join(active)}")
        return conflicts
    
    def resolve_conflicts(self, features_dict: Dict[str, int], text: str) -> Dict[str, int]:
        """Resolve feature conflicts using priority rules"""
        resolved = features_dict.copy()
        
        for group in self.strict_mutex_groups:
            active = [f for f in group if features_dict.get(f, 0) == 1]
            
            if len(active) > 1:
                # Define priority rules
                if group == ['attempt_prep', 'attempt_incomp', 'attempt_stop']:
                    priority = ['attempt_stop', 'attempt_incomp', 'attempt_prep']
                elif group == ['theft_pickpocket', 'theft_household']:
                    if any(kw in text for kw in ['室内', '房间', '卧室', '进入.*内']):
                        priority = ['theft_household', 'theft_pickpocket']
                    else:
                        priority = ['theft_pickpocket', 'theft_household']
                else:
                    priority = active
                
                # Keep only highest priority feature
                for feat in active:
                    resolved[feat] = 0
                
                for priority_feat in priority:
                    if priority_feat in active:
                        resolved[priority_feat] = 1
                        break
        
        return resolved
        
    def extract_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract legal features from texts with conflict resolution"""
        features = []
        feature_names = list(self.core_features.keys())
        
        for i, text in enumerate(texts):
            cleaned = self.preprocess_text(text)
            text_features = self._smart_extraction(cleaned)
            
            # Detect and resolve conflicts
            conflicts = self.detect_conflicts(text_features)
            if conflicts:
                text_features = self.resolve_conflicts(text_features, cleaned)
                
            feature_vector = [text_features[name] for name in feature_names]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32), feature_names
    
    def _smart_extraction(self, text: str) -> Dict[str, int]:
        """Extract features with context validation"""
        features = {}
        
        for feature_name, keywords in self.core_features.items():
            features[feature_name] = 0
            
            for keyword in keywords:
                if '.*' in keyword or '[' in keyword:
                    match = re.search(keyword, text)
                    if match and self._validate_context(feature_name, keyword, text):
                        features[feature_name] = 1
                            break
                else:
                    if keyword in text and self._validate_context(feature_name, keyword, text):
                        features[feature_name] = 1
                            break
        
        return features
    
    def _validate_context(self, feature_name: str, keyword: str, text: str) -> bool:
        """Validate feature in context to avoid false positives"""
        # Find keyword position
        if '.*' in keyword or '[' in keyword:
            match = re.search(keyword, text)
            if not match:
                return False
            pos = match.start()
        else:
            pos = text.find(keyword)
            if pos == -1:
                return False
        
        # Get context window (±50 chars)
        context = text[max(0, pos-50):min(len(text), pos+100)]
        
        # Validation rules for specific features
        if feature_name.startswith('attempt_'):
            crime_indicators = ['盗窃', '犯', '罪', '本院认为', '经审理查明', '被告人']
            legal_articles = ['第.*条', '处三年以下', '可以从轻']
            has_crime = any(ind in context for ind in crime_indicators)
            is_legal_article = any(art in context for art in legal_articles)
            return has_crime and not is_legal_article
            
        if feature_name.startswith('amount_'):
            amount_indicators = ['元', '价值', '金额', '财物', '数额', '人民币']
            pure_legal = ['处三年以下', '处十年以上', '法条全文']
            has_amount = any(ind in context for ind in amount_indicators)
            is_pure_legal = any(legal in context for legal in pure_legal)
            return has_amount and not is_pure_legal
            
        return True


class TheftSentencingPredictor:
    """ML-based theft crime sentencing predictor using legal features"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.feature_extractor = LegalFeatureExtractor()
        
        self.raw_df = None
        self.feature_matrix = None
        self.feature_names = None
        
        self.model = None
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'validation_split': 1/3,
            'test_split': 0.2,
            'random_state': 42,
            'output_dir': 'output/theft_prediction',
        }
    
    def load_data(self, file_path: str, text_col: str = '全文', 
                  target_col: str = '刑期', amount_col: str = '盗窃金额'):
        """Load and preprocess data"""
        print(f"\n[1/5] Loading data: {file_path}")
        
        self.raw_df = pd.read_csv(file_path)
        print(f"   Loaded {len(self.raw_df):,} records")
        
        # Basic cleaning
        self.raw_df = self.raw_df.dropna(subset=[text_col, target_col])
        self.raw_df = self.raw_df[self.raw_df[text_col].str.len() > 10]
        
        # Process theft amount
        self.raw_df[amount_col] = pd.to_numeric(self.raw_df[amount_col], errors='coerce').fillna(0)
        
        # Filter valid sentences (0-300 months)
        self.raw_df = self.raw_df[
            (self.raw_df[target_col] >= 0) & 
            (self.raw_df[target_col] <= 300)
        ]
        
        # Remove multi-defendant cases
        texts = self.raw_df[text_col].fillna('').astype(str)
        multi_defendant = (
            texts.str.contains('主犯', na=False) |
            texts.str.contains('从犯', na=False) |
            texts.str.contains('胁从犯', na=False)
        )
        self.raw_df = self.raw_df[~multi_defendant]
        
        print(f"   Cleaned data: {len(self.raw_df):,} valid records")
        
        self.text_col = text_col
        self.target_col = target_col
        self.amount_col = amount_col
        
        return self.raw_df
    
    def extract_features(self):
        """Extract legal features and theft amount"""
        print(f"\n[2/5] Extracting legal features")
        
        texts = self.raw_df[self.text_col].fillna('').astype(str).tolist()
        
        # Extract legal features
        legal_features, legal_names = self.feature_extractor.extract_features(texts)
        print(f"   Legal features: {len(legal_names)}")
        
        # Process theft amount (log transformation + normalization)
        amounts = self.raw_df[self.amount_col].values
        log_amounts = np.log1p(amounts)
        normalized_amounts = self.scaler.fit_transform(log_amounts.reshape(-1, 1)).flatten()
        
        # Combine features
        self.feature_matrix = np.column_stack([legal_features, normalized_amounts])
        self.feature_names = legal_names + ['theft_amount_normalized']
        
        # Normalize to [0, 1]
        self.feature_matrix = self.normalizer.fit_transform(self.feature_matrix)
        
        print(f"   Total features: {len(self.feature_names)}")
        
        return self.feature_matrix, self.feature_names
    
    def split_data(self):
        """Split data: 1/3 validation + 2/3 train/test"""
        print(f"\n[3/5] Splitting data")
        
        y = self.raw_df[self.target_col].values
        case_ids = self.raw_df['序号'].values if '序号' in self.raw_df.columns else self.raw_df.index.values
        
        # First split: validation vs train_test
        X_train_test, X_val, y_train_test, y_val, ids_train_test, ids_val = train_test_split(
            self.feature_matrix, y, case_ids,
            test_size=self.config['validation_split'],
            random_state=self.config['random_state']
        )
        
        # Second split: train vs test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_test, y_train_test,
            test_size=self.config['test_split'],
            random_state=self.config['random_state']
        )
        
        self.data_splits = {
            'X_train': X_train,
            'X_test': X_test,
            'X_val': X_val,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val,
            'ids_train_test': ids_train_test,
            'ids_val': ids_val
        }
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}, Validation: {len(X_val):,}")
        
        return self.data_splits
    
    def train_model(self):
        """Train XGBoost regression model"""
        print(f"\n[4/5] Training model")
        
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        X_test = self.data_splits['X_test']
        y_test = self.data_splits['y_test']
        
        # XGBoost model optimized for CAIL2018 metric
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
            reg_alpha=0.05,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
        )
        
            start_time = datetime.now()
        self.model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
            
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        cail_score = self._calculate_cail2018_score(y_test, y_pred)
            
        self.metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cail_score': cail_score,
            'train_time': train_time
        }
        
        print(f"   Test set - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, CAIL: {cail_score:.1f}%")
        
        return self.model
    
    def _calculate_cail2018_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate CAIL2018 competition score"""
        def score_func(v):
            if v <= 0.2: return 1.0
            elif v <= 0.4: return 0.8
            elif v <= 0.6: return 0.6
            elif v <= 0.8: return 0.4
            elif v <= 1.0: return 0.2
            else: return 0.0
        
        differences = []
        for t, p in zip(y_true, y_pred):
                t_safe = max(0.1, float(t))
                p_safe = max(0.1, float(p))
                diff = abs(math.log(t_safe + 1) - math.log(p_safe + 1))
                differences.append(diff)
        
        score = sum(score_func(d) for d in differences) / len(differences)
        return score * 100
    
    def predict_and_evaluate(self):
        """Generate predictions on validation set"""
        print(f"\n[5/5] Generating predictions")
        
        X_val = self.data_splits['X_val']
        y_val = self.data_splits['y_val']
        
        y_pred = self.model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 120)  # Clip predictions to reasonable range
        
        # Validation metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        cail_score = self._calculate_cail2018_score(y_val, y_pred)
        
        print(f"   Validation - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, CAIL: {cail_score:.1f}%")
        
        self.val_metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cail_score': cail_score
        }
        
        return y_pred, self.val_metrics
    
    def save_results(self, y_pred):
        """Save predictions and model"""
        print(f"\nSaving results")
        
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Validation predictions
        val_df = pd.DataFrame({
            'case_id': self.data_splits['ids_val'],
            'actual_sentence': self.data_splits['y_val'],
            'predicted_sentence': np.round(y_pred, 2),
            'absolute_error': np.round(np.abs(self.data_splits['y_val'] - y_pred), 2)
        })
        
        val_path = os.path.join(output_dir, 'validation_predictions.csv')
        val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
        print(f"   Predictions saved: {val_path}")
        
        # Save model
        model_path = os.path.join(output_dir, 'theft_model.joblib')
        joblib.dump(self.model, model_path)
        
        scaler_path = os.path.join(output_dir, 'normalizer.joblib')
        joblib.dump(self.normalizer, scaler_path)
        
        print(f"   Model saved: {model_path}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"   Feature importance saved: {importance_path}")
        
        return output_dir
    
    def run_pipeline(self, input_file: str, text_col: str = '全文', 
                     target_col: str = '刑期', amount_col: str = '盗窃金额'):
        """Execute complete prediction pipeline"""
        print("\nTheft Sentencing Prediction System")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Pipeline execution
            self.load_data(input_file, text_col, target_col, amount_col)
            self.extract_features()
            self.split_data()
            self.train_model()
            y_pred, val_metrics = self.predict_and_evaluate()
            output_dir = self.save_results(y_pred)
            
            # Summary
            elapsed = datetime.now() - start_time
            print(f"\n" + "=" * 60)
            print(f"Pipeline completed in {elapsed}")
            print(f"Samples: {len(self.raw_df):,}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Validation R²: {val_metrics['r2']:.3f}")
            print(f"CAIL2018 Score: {val_metrics['cail_score']:.1f}%")
            print(f"Output directory: {output_dir}")
            print("=" * 60)
            
            return output_dir
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Theft Sentencing Prediction System')
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='output/theft_prediction', help='Output directory')
    parser.add_argument('--text_column', type=str, default='全文', help='Text column name')
    parser.add_argument('--target_column', type=str, default='刑期', help='Target column name')
    parser.add_argument('--amount_column', type=str, default='盗窃金额', help='Amount column name')
    
    args = parser.parse_args()
    
    config = {
        'validation_split': 1/3,
        'test_split': 0.2,
        'random_state': 42,
        'output_dir': args.output_dir,
    }
    
    predictor = TheftSentencingPredictor(config)
    output_dir = predictor.run_pipeline(
        args.input_file, 
        args.text_column, 
        args.target_column,
        args.amount_column
    )
    
    if output_dir:
        print(f"\nSuccess! Results saved to: {output_dir}")
    else:
        print(f"\nPrediction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

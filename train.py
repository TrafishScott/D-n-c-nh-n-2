import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RecommendationSystem:
    def __init__(self):
        self.df = None
        self.results = {}
        self.evaluations = {}
        self.categorical_cols = ["indrel", "indrel_1mes", "tiprel_1mes", "indresi", "indext", "canal_entrada", "nomprov"]
        self.numerical_cols = ["age", "antiguedad"]
        self.target_cols = []
        self.feature_cols = []
        self.models_trained = False
        self.label_encoders = {}  # Lưu trữ các LabelEncoder đã huấn luyện
        
    def load_data(self, file_path):
        """Đọc dữ liệu từ file CSV"""
        try:
            self.df = pd.read_csv(file_path, low_memory=False)
            print(f"📋 Dữ liệu có {self.df.shape[0]} dòng và {self.df.shape[1]} cột")
            
            # Xác định các cột feature và target
            self.feature_cols = self.numerical_cols.copy()
            self.feature_cols.extend([col for col in self.categorical_cols if col in self.df.columns])
            self.target_cols = [col for col in self.df.columns if col.startswith("ind_") and col != "ind_empleado"]
            
            print(f"✅ Feature columns: {self.feature_cols}")
            print(f"✅ Target columns: {self.target_cols}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Xử lý dữ liệu trước khi huấn luyện và lưu các LabelEncoder"""
        # Xử lý cột phân loại
        for col in self.categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
                
                # Tạo và lưu LabelEncoder
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
                self.label_encoders[col] = encoder
        
        # Xử lý cột số
        for col in self.numerical_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        print("✅ Tiền xử lý dữ liệu hoàn tất!")
        
    def evaluate_model(self, model, X_test, y_test, target_name):
        """Đánh giá mô hình chi tiết"""
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Dự đoán xác suất
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # AUC-ROC
        auc_roc = None
        if y_pred_proba is not None:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # In thông tin đánh giá
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ CHO {target_name}")
        print("-" * 50)
        print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if auc_roc:
            print(f"AUC-ROC: {auc_roc:.4f}")
        
        # Trả về kết quả đánh giá
        evaluation = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        if auc_roc:
            evaluation['auc_roc'] = auc_roc
            
        return evaluation
    
    def train_models(self):
        """Huấn luyện mô hình cho từng sản phẩm"""
        if self.df is None:
            print("❌ Cần tải dữ liệu trước khi huấn luyện!")
            return False
        
        # Tiền xử lý dữ liệu
        self.preprocess_data()
        
        # Huấn luyện mô hình cho từng sản phẩm
        for target_col in self.target_cols:
            print(f"\n🔍 Đang huấn luyện mô hình cho sản phẩm: {target_col}")
            
            # Chỉ sử dụng các cột feature có trong dataset
            valid_features = [col for col in self.feature_cols if col in self.df.columns]
            
            # Đảm bảo không có giá trị NaN
            X = self.df[valid_features].copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].astype(str)
                    X[col] = LabelEncoder().fit_transform(X[col])
                elif X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())
            
            # Xử lý target
            if target_col in self.df.columns:
                y = self.df[target_col].copy()
                
                if y.dtype == 'object':
                    y = y.astype(str)
                    y = LabelEncoder().fit_transform(y)
                
                # Kiểm tra nếu chỉ có một lớp
                if len(y.unique()) == 1:
                    print(f"⚠️ Cột {target_col} chỉ có một giá trị {y.unique()[0]}. Bỏ qua!")
                    continue
                
                # Chia tập train - test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                try:
                    # Huấn luyện mô hình
                    model = XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        objective='binary:logistic',
                        eval_metric='logloss'
                    )
                    
                    model.fit(X_train, y_train)
                    
                    # Đánh giá mô hình
                    eval_results = self.evaluate_model(model, X_test, y_test, target_col)
                    
                    # Lưu kết quả
                    self.results[target_col] = {
                        "model": model,
                        "feature_importance": dict(zip(valid_features, model.feature_importances_))
                    }
                    
                    self.evaluations[target_col] = eval_results
                    
                except Exception as e:
                    print(f"❌ Lỗi khi huấn luyện mô hình cho {target_col}: {str(e)}")
            else:
                print(f"❌ Không tìm thấy cột {target_col} trong dữ liệu!")
        
        # Hiển thị so sánh hiệu suất
        self.compare_models()
        
        self.models_trained = True
        print("\n✅ Huấn luyện hoàn tất!")
        return True
    
    def compare_models(self):
        """So sánh hiệu suất các mô hình"""
        if not self.evaluations:
            return
            
        print("\n📈 SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH")
        print("-" * 50)
        
        # Tạo bảng so sánh
        comparison = pd.DataFrame({
            'Sản phẩm': [],
            'Độ chính xác': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC-ROC': []
        })
        
        for target, eval_results in self.evaluations.items():
            new_row = pd.DataFrame({
                'Sản phẩm': [target],
                'Độ chính xác': [eval_results['accuracy']],
                'Precision': [eval_results['precision']],
                'Recall': [eval_results['recall']],
                'F1-Score': [eval_results['f1_score']]
            })
            
            if 'auc_roc' in eval_results:
                new_row['AUC-ROC'] = eval_results['auc_roc']
            else:
                new_row['AUC-ROC'] = None
            
            comparison = pd.concat([comparison, new_row], ignore_index=True)
        
        # Sắp xếp theo F1-Score
        comparison = comparison.sort_values('F1-Score', ascending=False)
        print(comparison.to_string(index=False))
        
        # Hiển thị đặc trưng quan trọng
        self.show_feature_importance()
    
    def show_feature_importance(self):
        """Hiển thị đặc trưng quan trọng nhất"""
        if not self.results:
            return
            
        print("\n📊 ĐẶC TRƯNG QUAN TRỌNG NHẤT TRÊN TẤT CẢ CÁC MÔ HÌNH")
        print("-" * 50)
        
        # Tính điểm quan trọng trung bình
        all_features = {}
        for target, result in self.results.items():
            feature_importance = result["feature_importance"]
            
            for feature, importance in feature_importance.items():
                if feature not in all_features:
                    all_features[feature] = []
                
                all_features[feature].append(importance)
        
        # Tính trung bình và độ lệch chuẩn
        feature_summary = pd.DataFrame({
            'Đặc trưng': [],
            'Điểm tầm quan trọng trung bình': [],
            'Độ lệch chuẩn': []
        })
        
        for feature, scores in all_features.items():
            new_row = pd.DataFrame({
                'Đặc trưng': [feature],
                'Điểm tầm quan trọng trung bình': [np.mean(scores)],
                'Độ lệch chuẩn': [np.std(scores)]
            })
            
            feature_summary = pd.concat([feature_summary, new_row], ignore_index=True)
        
        # Sắp xếp theo điểm trung bình
        feature_summary = feature_summary.sort_values('Điểm tầm quan trọng trung bình', ascending=False)
        print(feature_summary.to_string(index=False))
    
    def preprocess_customer_data(self, customer_data):
        """Tiền xử lý dữ liệu khách hàng mới"""
        # Xử lý cột phân loại
        for col in self.categorical_cols:
            if col in customer_data.columns:
                customer_data[col] = customer_data[col].astype(str)
                
                # Sử dụng LabelEncoder đã lưu
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    # Xử lý giá trị mới không có trong tập huấn luyện
                    for i, val in enumerate(customer_data[col]):
                        if val not in encoder.classes_:
                            print(f"⚠️ Phát hiện giá trị mới '{val}' trong cột '{col}'. Thay thế bằng giá trị phổ biến nhất.")
                            # Tìm giá trị phổ biến nhất trong tập huấn luyện
                            most_common_val = encoder.classes_[0]  # Mặc định lấy giá trị đầu tiên
                            customer_data.loc[i, col] = most_common_val
                    
                    # Áp dụng transform
                    customer_data[col] = encoder.transform(customer_data[col])
        
        # Xử lý cột số
        for col in self.numerical_cols:
            if col in customer_data.columns:
                customer_data[col] = pd.to_numeric(customer_data[col], errors='coerce')
                if customer_data[col].isnull().any() and col in self.df.columns:
                    customer_data[col] = customer_data[col].fillna(self.df[col].median())
        
        return customer_data
    
    def predict_for_customer(self, customer_data):
        """Dự đoán các sản phẩm phù hợp cho khách hàng mới"""
        if not self.models_trained:
            print("❌ Cần huấn luyện mô hình trước khi dự đoán!")
            return None
        
        # Tiền xử lý dữ liệu khách hàng
        customer_data = self.preprocess_customer_data(customer_data)
        
        # Dự đoán
        predictions = {}
        
        for product, result in self.results.items():
            model = result["model"]
            valid_features = list(result["feature_importance"].keys())
            
            # Kiểm tra đủ các cột feature
            has_all_features = all(feature in customer_data.columns for feature in valid_features)
            if not has_all_features:
                print(f"⚠️ Thiếu một số cột đặc trưng cho sản phẩm {product}")
                continue
            
            # Dự đoán
            try:
                X_customer = customer_data[valid_features]
                proba = model.predict_proba(X_customer)[0, 1]  # Xác suất lớp dương
                pred = 1 if proba >= 0.5 else 0
                
                predictions[product] = {
                    "prediction": pred,
                    "probability": proba
                }
            except Exception as e:
                print(f"❌ Lỗi khi dự đoán cho {product}: {str(e)}")
        
        # Chuyển kết quả thành DataFrame
        pred_df = pd.DataFrame({
            'Sản phẩm': [],
            'Dự đoán': [],
            'Xác suất (%)': []
        })
        
        for product, result in predictions.items():
            new_row = pd.DataFrame({
                'Sản phẩm': [product],
                'Dự đoán': [result['prediction']],
                'Xác suất (%)': [result['probability'] * 100]
            })
            
            pred_df = pd.concat([pred_df, new_row], ignore_index=True)
        
        # Sắp xếp theo xác suất giảm dần
        pred_df = pred_df.sort_values('Xác suất (%)', ascending=False)
        
        return pred_df
    
    def get_new_customer_data(self):
        """Nhập dữ liệu khách hàng mới từ người dùng"""
        print("\n📝 NHẬP THÔNG TIN KHÁCH HÀNG MỚI")
        print("-" * 50)
        
        customer_data = {}
        
        # Nhập dữ liệu số
        for col in self.numerical_cols:
            while True:
                try:
                    value = input(f"Nhập {col} (ví dụ: age = 45, antiguedad = 15): ")
                    customer_data[col] = float(value)
                    break
                except ValueError:
                    print("❌ Vui lòng nhập một số hợp lệ!")
        
        # Nhập dữ liệu phân loại
        for col in self.categorical_cols:
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                print(f"\nGiá trị đã biết cho {col}: {', '.join(encoder.classes_)}")
            
            value = input(f"Nhập {col} (ví dụ: \nindrel = 1\nindrel_1mes = 1\ntiprel_1mes = A\nindresi = S\nindext = N\ncanal_entrada = KAF\nnomprov = MADRID)\n\nNhập: ")
            customer_data[col] = value
        
        # Chuyển thành DataFrame
        df_customer = pd.DataFrame(customer_data, index=[0])
        
        return df_customer

    def print_unique_values(self):
        """In ra các giá trị duy nhất cho từng cột phân loại"""
        if self.df is None:
            print("❌ Cần tải dữ liệu trước!")
            return
            
        print("\n📊 CÁC GIÁ TRỊ DUY NHẤT TRONG CỘT PHÂN LOẠI")
        print("-" * 50)
        
        for col in self.categorical_cols:
            if col in self.df.columns:
                unique_values = self.df[col].astype(str).unique()
                print(f"{col}: {', '.join(unique_values)}")
        
        print("\n📊 THỐNG KÊ CÁC CỘT SỐ")
        print("-" * 50)
        
        for col in self.numerical_cols:
            if col in self.df.columns:
                stats = self.df[col].describe()
                print(f"{col}:")
                print(f"  Min: {stats['min']}")
                print(f"  Max: {stats['max']}")
                print(f"  Mean: {stats['mean']}")
                print(f"  Median: {stats['50%']}")

def display_menu():
    """Hiển thị menu chính"""
    print("\n" + "=" * 50)
    print("🌟 HỆ THỐNG KHUYẾN NGHỊ SẢN PHẨM 🌟".center(50))
    print("=" * 50)
    print("1. Huấn luyện mô hình")
    print("2. Dự đoán cho khách hàng mới")
    print("3. Hiển thị thông tin dữ liệu")
    print("4. Thoát")
    print("=" * 50)
    
    choice = input("Chọn chức năng (1-4): ")
    return choice

def main():
    """Hàm chính của chương trình"""
    system = RecommendationSystem()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            print("\n🔄 HUẤN LUYỆN MÔ HÌNH")
            print("-" * 50)
            
            # Nhập đường dẫn file dữ liệu
            file_path = input("Nhập đường dẫn file CSV (nhấn Enter để sử dụng đường dẫn mặc định): ")
            if not file_path:
                file_path = "test.csv"
            
            # Tải dữ liệu
            if system.load_data(file_path):
                # Huấn luyện mô hình
                system.train_models()
            
            input("\nNhấn Enter để tiếp tục...")
            
        elif choice == '2':
            if not system.models_trained:
                print("\n❌ Cần huấn luyện mô hình trước khi dự đoán!")
                input("\nNhấn Enter để tiếp tục...")
                continue
            
            print("\n🔮 DỰ ĐOÁN CHO KHÁCH HÀNG MỚI")
            print("-" * 50)
            
            # Nhập dữ liệu khách hàng mới
            customer_data = system.get_new_customer_data()
            
            # Dự đoán
            predictions = system.predict_for_customer(customer_data)
            
            # Hiển thị kết quả
            if predictions is not None:
                print("\n🎯 CÁC SẢN PHẨM PHÙ HỢP CHO KHÁCH HÀNG:")
                print(predictions)
                
                # Hiển thị các sản phẩm được khuyến nghị
                recommended = predictions[predictions['Xác suất (%)'] > 50]
                if not recommended.empty:
                    print("\n🔆 CÁC SẢN PHẨM ĐƯỢC KHUYẾN NGHỊ:")
                    print(recommended)
                else:
                    print("\n⚠️ Không có sản phẩm nào có xác suất cao (>50%)")
            
            input("\nNhấn Enter để tiếp tục...")
            
        elif choice == '3':
            if system.df is None:
                print("\n❌ Cần tải dữ liệu trước!")
                input("\nNhấn Enter để tiếp tục...")
                continue
                
            print("\n📊 THÔNG TIN DỮ LIỆU")
            system.print_unique_values()
            input("\nNhấn Enter để tiếp tục...")
            
        elif choice == '4':
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
            break
            
        else:
            print("\n❌ Lựa chọn không hợp lệ! Vui lòng chọn lại.")

if __name__ == "__main__":
    main()
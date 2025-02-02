from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
import matplotlib
matplotlib.use('Agg')  # GUI backend yerine Agg kullan

app = Flask(__name__)
CORS(app)  # Tüm originlere izin ver

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        app.logger.info("\n=== Analysis Starting ===")
        
        # Request body'sini al ve DataFrame oluştur
        data = request.get_json(force=True)
        df = pd.DataFrame(data)
        app.logger.info(f"DataFrame created: {df.shape}")
        
        # Veri tiplerini dönüştür
        df['transaction_qty'] = pd.to_numeric(df['transaction_qty'], errors='coerce')
        df['unit_price'] = df['unit_price'].str.replace(',', '.').astype(float)
        
        # Total sales hesapla (en başta)
        df["total_sales"] = df["transaction_qty"] * df["unit_price"]
        
        plots = []
        
        try:
            # ABC Analysis (önce ABC analizi)
            app.logger.info("ABC Analysis starting...")
            
            # ABC analizi için grupla
            df_abc = df.groupby("product_detail")["total_sales"].sum().reset_index()
            df_abc = df_abc.sort_values(by="total_sales", ascending=False)
            df_abc["cumulative_percentage"] = df_abc["total_sales"].cumsum() / df_abc["total_sales"].sum() * 100
            
            # ABC kategorilerini belirle
            def classify_abc(percentage):
                if percentage <= 70: return "A"
                elif percentage <= 90: return "B"
                else: return "C"
            
            df_abc["ABC_Category"] = df_abc["cumulative_percentage"].apply(classify_abc)
            
            # ABC grafiği oluştur
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_abc, x="ABC_Category", hue="ABC_Category", palette="coolwarm", legend=False)
            plt.title("ABC Analysis: Product Classification")
            plt.xlabel("ABC Category")
            plt.ylabel("Count of Products")
            
            # Grafiği kaydet
            plots.append({
                'image': fig_to_base64(plt.gcf()),
                'title': 'ABC Analysis',
                'description': (
                    "📊 Product Classification Summary:\n"
                    f"📌 Category A: {len(df_abc[df_abc.ABC_Category=='A'])} products - High value items\n"
                    f"📌 Category B: {len(df_abc[df_abc.ABC_Category=='B'])} products - Medium value items\n"
                    f"📌 Category C: {len(df_abc[df_abc.ABC_Category=='C'])} products - Low value items\n"
                    "\n📈 Recommendations:\n"
                    "• Focus inventory control on Category A items\n"
                    "• Implement periodic review for Category B items\n"
                    "• Use simplified control for Category C items"
                )
            })
            plt.close()
            
            app.logger.info("ABC Analysis completed")
            
        except Exception as e:
            app.logger.error(f"ABC Analysis error: {str(e)}")
            
        try:
            # Stok Analizi (ABC analizinden sonra)
            app.logger.info("Stock Analysis starting...")
            
            # Her ürün için son 7 günlük satış trendini hesapla
            df['datetime'] = pd.to_datetime(df['transaction_date'].astype(str))
            last_week = df['datetime'].max() - pd.Timedelta(days=7)
            
            df_stock = df[df['datetime'] > last_week].groupby('product_detail').agg({
                'transaction_qty': ['sum', 'mean', 'count'],
                'total_sales': ['sum', 'mean']
            }).reset_index()
            
            df_stock.columns = ['product_detail', 'weekly_qty', 'avg_daily_qty', 'transaction_count', 'weekly_sales', 'avg_daily_sales']
            
            # Kritik stok seviyesini belirle (günlük ortalama satışın 3 katı)
            df_stock['critical_stock'] = df_stock['avg_daily_qty'] * 3
            
            # A ve B kategorisindeki ürünleri filtrele
            critical_products = df_stock[
                (df_stock['product_detail'].isin(df_abc[df_abc['ABC_Category'].isin(['A', 'B'])]['product_detail'])) &
                (df_stock['weekly_qty'] > 0)
            ].sort_values('weekly_sales', ascending=False)
            
            # Top 10 kritik ürünü görselleştir
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=critical_products.head(10),
                x='weekly_qty',
                y='product_detail',
                palette='YlOrRd'
            )
            plt.title('Critical Stock Levels (Last 7 Days)')
            plt.xlabel('Weekly Sales Quantity')
            plt.ylabel('Product')
            
            # Uyarı mesajını oluştur
            warning_msg = (
                "⚠️ Critical Stock Alert\n\n"
                "📊 Top Priority Items:\n"
            )
            for _, row in critical_products.head(10).iterrows():
                warning_msg += (
                    f"\n🔍 {row['product_detail']}\n"
                    f"   • Weekly Sales Volume: {row['weekly_qty']:.0f} units\n"
                    f"   • Average Daily Sales: {row['avg_daily_qty']:.1f} units\n"
                    f"   • Recommended Stock Level: {row['critical_stock']:.0f} units\n"
                    f"   • Action Required: Immediate reorder\n"
                )
            
            # Grafiği kaydet
            plots.append({
                'image': fig_to_base64(plt.gcf()),
                'title': 'Stock Analysis and Order Recommendations',
                'description': warning_msg
            })
            plt.close()
            
            app.logger.info("Stock Analysis completed")
            
        except Exception as e:
            app.logger.error(f"Stock Analysis error: {str(e)}")
            
        try:
            # FRM Analysis
            app.logger.info("FRM Analysis starting...")
            
            # Datetime sütununu oluştur
            df["datetime"] = pd.to_datetime(df["transaction_date"].astype(str) + " " + df["transaction_time"])
            reference_datetime = df["datetime"].max() + pd.Timedelta(minutes=1)
            
            # FRM metriklerini hesapla
            df_frm = df.groupby("store_id").agg(
                Recency=("datetime", lambda x: (reference_datetime - x.max()).total_seconds() / 60),
                Frequency=("transaction_id", "count"),
                Monetary=("total_sales", "sum")
            ).reset_index()
            
            # FRM grafiği
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_frm, x="Recency", y="Monetary", size="Frequency", sizes=(20, 200))
            plt.title("FRM Analysis: Customer Segmentation")
            plt.xlabel("Recency (Minutes)")
            plt.ylabel("Monetary Value")
            
            # Grafiği kaydet
            plots.append({
                'image': fig_to_base64(plt.gcf()),
                'title': 'FRM Analysis',
                'description': (
                    "📊 Store Performance Metrics:\n\n"
                    f"⏱️ Recency\n"
                    f"   • Average: {df_frm.Recency.mean():.1f} minutes\n"
                    f"   • Best: {df_frm.Recency.min():.1f} minutes\n\n"
                    f"🔄 Frequency\n"
                    f"   • Average visits: {df_frm.Frequency.mean():.1f}\n"
                    f"   • Highest: {df_frm.Frequency.max():.0f}\n\n"
                    f"💰 Monetary\n"
                    f"   • Average spend: ${df_frm.Monetary.mean():.2f}\n"
                    f"   • Highest spend: ${df_frm.Monetary.max():.2f}"
                )
            })
            plt.close()
            
            app.logger.info("FRM Analysis completed")
            
        except Exception as e:
            app.logger.error(f"FRM Analysis error: {str(e)}")
            
        try:
            # Stok Durumu Analizi
            app.logger.info("Stock Status Analysis starting...")
            
            # Son 30 günlük satış trendini hesapla
            last_month = df['datetime'].max() - pd.Timedelta(days=30)
            df_last_month = df[df['datetime'] > last_month]
            
            # Ürün bazında satış istatistikleri
            df_stock_status = df_last_month.groupby('product_detail').agg({
                'transaction_qty': ['sum', 'mean', 'std'],
                'total_sales': 'sum'
            }).reset_index()
            
            # Sütun isimlerini düzenle
            df_stock_status.columns = ['product_detail', 'monthly_qty', 'daily_avg', 'daily_std', 'monthly_sales']
            
            # Güvenlik stok seviyesi hesapla (2 sigma - %95 güven aralığı)
            df_stock_status['safety_stock'] = (df_stock_status['daily_avg'] + 2 * df_stock_status['daily_std']) * 7  # 1 haftalık
            
            # Kritik ürünleri belirle (A ve B kategorisi ürünlerinden)
            critical_items = df_stock_status[
                df_stock_status['product_detail'].isin(
                    df_abc[df_abc['ABC_Category'].isin(['A', 'B'])]['product_detail']
                )
            ].sort_values('monthly_sales', ascending=False)
            
            # En kritik 15 ürünü görselleştir
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=critical_items.head(15),
                x='monthly_qty',
                y='product_detail',
                palette='RdYlGn_r'  # Kırmızı-Sarı-Yeşil renk paleti (tersten)
            )
            plt.title('Kritik Stok Durumu (Son 30 Gün)')
            plt.xlabel('Aylık Satış Miktarı')
            plt.ylabel('Ürün')
            
            # Detaylı stok raporu oluştur
            stock_report = (
                "📊 Inventory Status Report\n\n"
                "🚨 Critical Items Requiring Attention:\n"
            )
            
            for _, row in critical_items.head(15).iterrows():
                safety_level = row['safety_stock']
                current_level = row['monthly_qty'] / 30
                
                if current_level < safety_level:
                    risk_level = "⚠️ HIGH RISK" if current_level < safety_level/2 else "⚡ MEDIUM RISK"
                    stock_report += (
                        f"\n{risk_level} {row['product_detail']}\n"
                        f"   • Daily Sales: {row['daily_avg']:.1f} units\n"
                        f"   • Safety Stock: {safety_level:.0f} units\n"
                        f"   • Monthly Volume: {row['monthly_qty']:.0f} units\n"
                        f"   • Revenue Impact: ${row['monthly_sales']:.2f}\n"
                    )
            
            # Grafiği kaydet
            plots.append({
                'image': fig_to_base64(plt.gcf()),
                'title': 'Stock Status Analysis',
                'description': stock_report
            })
            plt.close()
            
            app.logger.info("Stock Status Analysis completed")
            
        except Exception as e:
            app.logger.error(f"Stock Status Analysis error: {str(e)}")
            
        # Sonuç döndür
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed',
            'data_shape': df.shape,
            'plots': plots
        })

    except Exception as e:
        app.logger.error(f"ERROR: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
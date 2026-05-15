import os
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analysis 2018",
    page_icon="📊",
    layout="wide",
)

# Load dan proses data
@st.cache_data
def load_data():
    BASE = os.path.join(os.path.dirname(__file__), "data")

    # 1. Load semua dataset
    customers_df          = pd.read_csv(f"{BASE}/customers_dataset.csv")
    orders_df             = pd.read_csv(f"{BASE}/orders_dataset.csv")
    order_items_df        = pd.read_csv(f"{BASE}/order_items_dataset.csv")
    order_reviews_df      = pd.read_csv(f"{BASE}/order_reviews_dataset.csv")
    products_df           = pd.read_csv(f"{BASE}/products_dataset.csv")
    products_translation_df = pd.read_csv(f"{BASE}/product_category_name_translation.csv")

    # 2. Cleaning — fix tipe data datetime
    datetime_columns = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in datetime_columns:
        orders_df[col] = pd.to_datetime(orders_df[col])

    # 3. Cleaning — fix missing value
    products_df["product_category_name"] = (
        products_df["product_category_name"].fillna("unknown")
    )

    # 4. Cleaning — fix duplicate (1 review per order)
    df_reviews_clean = (
        order_reviews_df
        .sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
        .reset_index(drop=True)
    )

    # 5. Cleaning — fix outlier (winsorization IQR)
    for col in ["price", "freight_value"]:
        Q1  = order_items_df[col].quantile(0.25)
        Q3  = order_items_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(Q1 - 1.5 * IQR, 0)
        upper = Q3 + 1.5 * IQR
        order_items_df[col] = order_items_df[col].clip(lower=lower, upper=upper)

    # 6. Filter orders 2018 delivered
    orders_df_2018 = orders_df[
        (orders_df["order_status"] == "delivered") &
        (orders_df["order_purchase_timestamp"].dt.year == 2018)
    ].copy()

    # 7. Build master_df
    master_df = (
        orders_df_2018
        .merge(order_items_df, on="order_id", how="left")
        .merge(products_df[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(products_translation_df, on="product_category_name", how="left")
        .merge(df_reviews_clean[["order_id", "review_score"]], on="order_id", how="left")
        .merge(customers_df[["customer_id", "customer_unique_id",
                              "customer_city", "customer_state"]], on="customer_id", how="left")
    )
    master_df["revenue"] = master_df["price"] + master_df["freight_value"]
    master_df["product_category_name_english"] = (
        master_df["product_category_name_english"]
        .fillna(master_df["product_category_name"])
    )

    # EDA Q1
    category_df = (
        master_df
        .groupby("product_category_name_english", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            avg_review_score=("review_score", "mean"),
            total_orders=("order_id", "nunique"),
            total_items=("order_item_id", "count"),
        )
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    category_problem_df = (
        category_df[category_df["avg_review_score"] < 4]
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    # EDA Q2
    ref_date = orders_df_2018["order_purchase_timestamp"].max()

    rfm_df_base = (
        orders_df_2018
        .merge(customers_df[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
        .groupby("customer_unique_id", as_index=False)
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
        )
    )
    rfm_df_base["recency_days"] = (ref_date - rfm_df_base["last_purchase"]).dt.days
    rfm_df_base["churn_risk"]   = (
        (rfm_df_base["recency_days"] > 90) & (rfm_df_base["frequency"] < 2)
    )

    # RFM Analysis
    monetary_df = (
        master_df
        .groupby("customer_unique_id", as_index=False)
        .agg(monetary=("revenue", "sum"))
    )
    rfm_df = rfm_df_base.merge(monetary_df, on="customer_unique_id", how="left")

    # Scoring
    rfm_df["r_score"] = pd.qcut(
        rfm_df["recency_days"], q=4, labels=[4, 3, 2, 1]
    ).astype(int)
    rfm_df["f_score"] = pd.cut(
        rfm_df["frequency"],
        bins=[0, 1, 2, 3, rfm_df["frequency"].max()],
        labels=[1, 2, 3, 4],
    ).astype(int)
    rfm_df["m_score"] = pd.qcut(
        rfm_df["monetary"], q=4, labels=[1, 2, 3, 4]
    ).astype(int)
    rfm_df["rfm_score"] = rfm_df["r_score"] + rfm_df["f_score"] + rfm_df["m_score"]

    # Segmentasi (sesuai notebook)
    def segment_rfm(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 3 and f >= 3:               return "Pelanggan Terbaik"
        elif r >= 3 and f == 2:             return "Pelanggan Setia"
        elif r >= 3 and f == 1 and m >= 3:  return "Pelanggan Potensial"
        elif r >= 3 and f == 1 and m < 3:   return "Pelanggan Baru"
        elif r == 2 and f >= 2:             return "Perlu Perhatian"
        elif r == 2 and f == 1:             return "Pelanggan Tidak Aktif"
        elif r == 1 and f >= 2:             return "Berisiko Churn"
        else:                               return "Pelanggan Hilang"

    rfm_df["segment"] = rfm_df.apply(segment_rfm, axis=1)

    rfm_summary = (
        rfm_df
        .groupby("segment", as_index=False)
        .agg(
            jumlah_pelanggan=("customer_unique_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        .sort_values("jumlah_pelanggan", ascending=False)
        .reset_index(drop=True)
    )

    return (
        master_df, orders_df_2018,
        category_df, category_problem_df,
        rfm_df_base, rfm_df, rfm_summary,
    )


# Load data
with st.spinner("Memuat dan memproses data..."):
    (
        master_df, orders_df_2018,
        category_df, category_problem_df,
        rfm_df_base, rfm_df, rfm_summary,
    ) = load_data()

# Konstanta warna segmen (sesuai notebook)
PALETTE = {
    "Pelanggan Terbaik"     : "#2ecc71",
    "Pelanggan Setia"       : "#27ae60",
    "Pelanggan Potensial"   : "#3498db",
    "Pelanggan Baru"        : "#85c1e9",
    "Perlu Perhatian"       : "#f39c12",
    "Pelanggan Tidak Aktif" : "#e67e22",
    "Berisiko Churn"        : "#e74c3c",
    "Pelanggan Hilang"      : "#922b21",
}


# Header dan KPI
st.title("E-Commerce Public Dataset Analysis")
st.caption(
    "Periode: Januari – Desember 2018  ·  Status Order: Delivered"
)
st.divider()

churn_total = int(rfm_df_base["churn_risk"].sum())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Orders",       f"{orders_df_2018['order_id'].nunique():,}")
c2.metric("Unique Customers",   f"{rfm_df_base['customer_unique_id'].nunique():,}")
c3.metric("Total Revenue",      f"R${master_df['revenue'].sum():,.0f}")
c4.metric("Avg Review Score",   f"{master_df['review_score'].mean():.2f} / 5")
c5.metric("Churn Risk",         f"{churn_total:,}",
          f"{churn_total/len(rfm_df_base)*100:.1f}% dari total pelanggan")

st.divider()


# Tabs
tab1, tab2, tab3 = st.tabs([
    "Question 1: Revenue & Review",
    "Question 2: Churn Risk",
    "RFM Analysis",
])


# Tab 1 - Q1
with tab1:
    st.subheader("Kategori Produk yang Memiliki Revenue Tertinggi dengan Rata-rata Review < 4")
    st.markdown(
        "> Mengidentifikasi kategori produk yang menghasilkan revenue besar "
        "namun memiliki tingkat kepuasan pelanggan rendah (avg review score < 4) "
        "selama periode **Januari – Desember 2018**."
    )

    top_n = st.slider("Tampilkan Top-N kategori", min_value=5, max_value=20, value=10, step=1)
    st.divider()

    plot_df = category_problem_df.head(top_n).copy()
    plot_df["category_label"] = (
        plot_df["product_category_name_english"]
        .str.replace("_", " ").str.title()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.6)))
    fig.suptitle(
        f"Top {top_n} Kategori: Revenue Tertinggi & Rata-rata Review < 4 (2018)",
        fontsize=13, fontweight="bold",
    )

    rev_colors = [
        "#922b21" if s < 3.5 else "#e67e22" if s < 3.75 else "#f4d03f"
        for s in plot_df["avg_review_score"][::-1]
    ]

    # Panel kiri - Total Revenue
    bars1 = axes[0].barh(
        plot_df["category_label"][::-1],
        plot_df["total_revenue"][::-1] / 1e3,
        color=rev_colors,
    )
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(
            w + 1, bar.get_y() + bar.get_height() / 2,
            f"R${w:,.0f}K", va="center", fontsize=9,
        )
    axes[0].set_xlabel("Total Revenue (R$ Thousand)")
    axes[0].set_title("Total Revenue", fontweight="bold")
    axes[0].set_xlim(0, plot_df["total_revenue"].max() / 1e3 * 1.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Panel kanan - Avg Review Score
    bars2 = axes[1].barh(
        plot_df["category_label"][::-1],
        plot_df["avg_review_score"][::-1],
        color=rev_colors,
    )
    axes[1].axvline(4, color="red", linestyle="--", lw=1.5, label="Threshold = 4.0")
    for bar, val in zip(bars2, plot_df["avg_review_score"][::-1]):
        axes[1].text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9,
        )
    axes[1].set_xlabel("Rata-rata Skor Review (1–5)")
    axes[1].set_title("Avg Review Score", fontweight="bold")
    axes[1].set_xlim(0, 5)
    axes[1].set_yticklabels([])
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    legend_items = [
        mpatches.Patch(color="#922b21", label="Review < 3.50"),
        mpatches.Patch(color="#e67e22", label="Review 3.50 – 3.74"),
        mpatches.Patch(color="#f4d03f", label="Review 3.75 – 3.99"),
    ]
    axes[0].legend(handles=legend_items, fontsize=8, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Tabel ringkasan
    st.divider()
    st.markdown("##### Tabel Ringkasan")
    tbl1 = plot_df[["category_label", "total_revenue", "avg_review_score", "total_orders"]].copy()
    tbl1.columns = ["Kategori", "Total Revenue (R$)", "Avg Review Score", "Total Orders"]
    tbl1["Total Revenue (R$)"]  = tbl1["Total Revenue (R$)"].apply(lambda x: f"R${x:,.2f}")
    tbl1["Avg Review Score"]    = tbl1["Avg Review Score"].apply(lambda x: f"{x:.2f}")
    tbl1["Total Orders"]        = tbl1["Total Orders"].apply(lambda x: f"{x:,}")
    st.dataframe(tbl1, use_container_width=True, hide_index=True)

    st.info(
        "**💡 Kesimpulan:** Terdapat 5 kategori produk dengan revenue > R$170.000 "
        "namun rata-rata skor review di bawah 4, yaitu *Bed Bath Table*, "
        "*Computers Accessories*, *Furniture Decor*, *Telephony*, dan *Office Furniture*. "
        "Ini mengindikasikan gap antara volume penjualan dan kepuasan pelanggan."
    )


# Tab 2 - Q2
with tab2:
    st.subheader("Analisis Pelanggan Churn Risk")
    st.markdown(
        "> Pelanggan dikategorikan **churn risk** apabila memenuhi **dua kriteria** berikut:  \n"
        "> - Recency > 90 hari (tidak bertransaksi dalam 90 hari terakhir)  \n"
        "> - Frequency < 2 transaksi selama periode **Januari – Desember 2018**"
    )
    st.divider()

    total_cust  = len(rfm_df_base)
    churn_count = int(rfm_df_base["churn_risk"].sum())
    non_churn   = total_cust - churn_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pelanggan",  f"{total_cust:,}")
    c2.metric("Churn Risk",       f"{churn_count:,}",
              f"{churn_count/total_cust*100:.1f}% dari total")
    c3.metric("Non-Churn Risk",   f"{non_churn:,}",
              f"{non_churn/total_cust*100:.1f}% dari total")

    col_l, col_r = st.columns(2)

    # Pie chart
    with col_l:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [non_churn, churn_count],
            labels=["Non-Churn Risk", "Churn Risk"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=140,
            explode=(0, 0.06),
            textprops={"fontsize": 11},
            pctdistance=0.78,
        )
        ax.set_title(
            f"Proporsi Churn Risk\n(Total: {total_cust:,} pelanggan)",
            fontsize=11,
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Scatter recency vs frequency
    with col_r:
        sample = rfm_df_base.sample(min(3000, len(rfm_df_base)), random_state=42)
        c_list = ["#e74c3c" if v else "#3498db" for v in sample["churn_risk"]]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            sample["recency_days"], sample["frequency"],
            c=c_list, alpha=0.4, s=15, edgecolors="none",
        )
        ax.axvline(90, color="red",    linestyle="--", lw=1.5)
        ax.axhline(2,  color="orange", linestyle="--", lw=1.5)

        legend_els = [
            mpatches.Patch(color="#e74c3c", label=f"Churn Risk ({churn_count:,})"),
            mpatches.Patch(color="#3498db", label=f"Non-Churn Risk ({non_churn:,})"),
            plt.Line2D([0], [0], color="red",    linestyle="--", label="Recency = 90 hari"),
            plt.Line2D([0], [0], color="orange", linestyle="--", label="Frequency = 2"),
        ]
        ax.legend(handles=legend_els, fontsize=8, loc="upper right")
        ax.set_xlabel("Recency (Hari Sejak Pembelian Terakhir)")
        ax.set_ylabel("Frequency (Jumlah Transaksi)")
        ax.set_title("Recency vs Frequency", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Distribusi recency
    st.divider()
    st.markdown("##### Distribusi Recency Pelanggan")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.hist(
        rfm_df_base.loc[~rfm_df_base["churn_risk"], "recency_days"],
        bins=40, color="#3498db", alpha=0.7, label="Non-Churn Risk",
    )
    ax.hist(
        rfm_df_base.loc[rfm_df_base["churn_risk"], "recency_days"],
        bins=40, color="#e74c3c", alpha=0.7, label="Churn Risk",
    )
    ax.axvline(90, color="black", linestyle="--", lw=1.5, label="Threshold 90 hari")
    ax.set_xlabel("Recency (Hari)")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.warning(
        "**💡 Kesimpulan:** Terdapat **32.376 pelanggan** (62,7%) yang masuk kategori "
        "churn risk. Mayoritas pelanggan hanya melakukan 1 transaksi dan tidak kembali "
        "dalam 90 hari — mengindikasikan rendahnya loyalitas pelanggan."
    )


# Tab 3 - RFM
with tab3:
    st.subheader("RFM Analysis – Segmentasi Pelanggan 2018")
    st.markdown(
        "Pelanggan dikelompokkan ke dalam **8 segmen** menggunakan pendekatan "
        "**rule-based binning** berdasarkan nilai Recency, Frequency, dan Monetary."
    )
    st.divider()

    # KPI segmen utama
    def get_seg_count(name):
        row = rfm_summary[rfm_summary["segment"] == name]
        return int(row["jumlah_pelanggan"].values[0]) if len(row) else 0

    total_rfm  = rfm_df["customer_unique_id"].nunique()
    terbaik    = get_seg_count("Pelanggan Terbaik")
    hilang     = get_seg_count("Pelanggan Hilang")
    potensial  = get_seg_count("Pelanggan Potensial")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pelanggan",      f"{total_rfm:,}")
    c2.metric("Pelanggan Terbaik",    f"{terbaik:,}",
              f"{terbaik/total_rfm*100:.1f}%")
    c3.metric("Pelanggan Potensial",  f"{potensial:,}",
              f"{potensial/total_rfm*100:.1f}%")
    c4.metric("Pelanggan Hilang",     f"{hilang:,}",
              f"{hilang/total_rfm*100:.1f}%")
    st.divider()

    segment_order = (rfm_summary
                     .sort_values("jumlah_pelanggan")["segment"].tolist())
    colors        = [PALETTE.get(s, "#95a5a6") for s in segment_order]
    plot_data     = rfm_summary.set_index("segment").reindex(segment_order)

    col_l, col_r = st.columns(2)

    # Jumlah pelanggan per segmen
    with col_l:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(segment_order, plot_data["jumlah_pelanggan"], color=colors)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 30, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9)
        ax.set_xlabel("Jumlah Pelanggan")
        ax.set_title("Distribusi Segmen Pelanggan", fontweight="bold")
        ax.set_xlim(0, plot_data["jumlah_pelanggan"].max() * 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Avg monetary per segmen
    with col_r:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars2 = ax.barh(segment_order, plot_data["avg_monetary"], color=colors)
        for bar in bars2:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                    f"R${w:,.0f}", va="center", fontsize=9)
        ax.set_xlabel("Rata-rata Total Belanja (R$)")
        ax.set_title("Rata-rata Nilai Belanja per Segmen", fontweight="bold")
        ax.set_xlim(0, plot_data["avg_monetary"].max() * 1.3)
        ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tabel ringkasan RFM
    st.divider()
    st.markdown("##### Tabel Ringkasan Segmen RFM")
    tbl_rfm = rfm_summary.copy()
    tbl_rfm.columns = [
        "Segmen", "Jumlah Pelanggan",
        "Avg Recency (Hari)", "Avg Frequency",
        "Avg Monetary (R$)", "Total Monetary (R$)",
    ]
    tbl_rfm["Jumlah Pelanggan"]    = tbl_rfm["Jumlah Pelanggan"].apply(lambda x: f"{x:,}")
    tbl_rfm["Avg Recency (Hari)"]  = tbl_rfm["Avg Recency (Hari)"].apply(lambda x: f"{x:.1f}")
    tbl_rfm["Avg Frequency"]       = tbl_rfm["Avg Frequency"].apply(lambda x: f"{x:.2f}")
    tbl_rfm["Avg Monetary (R$)"]   = tbl_rfm["Avg Monetary (R$)"].apply(lambda x: f"R${x:,.2f}")
    tbl_rfm["Total Monetary (R$)"] = tbl_rfm["Total Monetary (R$)"].apply(lambda x: f"R${x:,.2f}")
    st.dataframe(tbl_rfm, use_container_width=True, hide_index=True)

    # Legenda warna segmen
    st.divider()
    st.markdown("##### Keterangan Segmen")
    keterangan = {
        "Pelanggan Terbaik":     "Recency tinggi, frekuensi tinggi — pelanggan paling loyal dan berharga.",
        "Pelanggan Setia":       "Recency tinggi, frekuensi sedang — pelanggan yang mulai menunjukkan loyalitas.",
        "Pelanggan Potensial":   "Recency tinggi, monetary tinggi — berpotensi menjadi pelanggan terbaik.",
        "Pelanggan Baru":        "Baru pertama kali bertransaksi dengan nilai belanja rendah.",
        "Perlu Perhatian":       "Recency sedang, frekuensi sedang — perlu didekati sebelum berhenti.",
        "Pelanggan Tidak Aktif": "Recency sedang, frekuensi rendah — mulai menunjukkan tanda tidak aktif.",
        "Berisiko Churn":        "Recency rendah, frekuensi tinggi — pernah aktif namun mulai menghilang.",
        "Pelanggan Hilang":      "Recency rendah, frekuensi rendah — kemungkinan besar sudah berhenti.",
    }
    for seg, desc in keterangan.items():
        st.markdown(f"- **{seg}**: {desc}")

    st.success(
        "**💡 Kesimpulan:** Segmen *Pelanggan Terbaik* memiliki nilai belanja tertinggi "
        "dan harus diprioritaskan dalam program retensi VIP. "
        "Segmen *Pelanggan Hilang* mendominasi jumlah pelanggan — konsisten dengan temuan "
        "churn risk 62,7% pada Q2. "
        "Segmen *Pelanggan Baru* dan *Pelanggan Potensial* adalah peluang terbesar "
        "untuk ditingkatkan melalui kampanye engagement yang tepat."
    )


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# === Mapping Kata Kunci ke Aspek ===
aspect_keywords = {
    'Lagu': [
        'song', 'songs', 'play', 'cant', 'music',
        'like', 'fun', 'playing', 'amazing',
        'beginner', 'beginners', 'hard',
    ],
    'Harga': [
        'premium', 'pay', 'free', 'purchase', 'money', 'price',
        'subscribe', 'subscription', 'worth', 'ads', 'trial',
        'charged', 'payment'
    ],
    'Tutorial': [
        'learn', 'learning', 'lessons', 'helpful', 'love', 'instructor',
        'tutorial', 'teaching', 'great', 'good', 'best', 'easy',
        'amazing', 'guide', 'gamification'
    ],
    'Login': [
        'login', 'account', 'log', 'sign', 'sign in', 'sign up',
        'cant login', 'log in', 'register', 'access', 'out',
        'auth', 'reset', 'email', 'password'
    ],
    'Teknis': [
        'tuning', 'tune', 'sound', 'mode', 'time', 'try', 'get', 'frustrating',
        'crash', 'bug', 'glitch', 'slow', 'lag', 'freeze', 'string',
        'fail', 'issue', 'update', 'load', 'problem', 'error', 'close'
    ]
}

def extract_aspect(text):
    text = str(text).lower()
    for aspect, keywords in aspect_keywords.items():
        if any(kw in text for kw in keywords):
            return aspect
    return None

def map_sentiment(label):
    if label.lower() == 'positive':
        return 'Positif'
    elif label.lower() == 'negative':
        return 'Negatif'
    return None

st.set_page_config(page_title="🎵 Sentiment App – Ukulele by Yousician")
st.title("🎵 Sentiment App – Ukulele by Yousician")

st.write("Unggah file CSV hasil prediksi untuk dilakukan analisis aspek dan visualisasi sentimen.")

uploaded_file = st.file_uploader("📁 Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ekstraksi aspek dan label sentimen
    df['aspect'] = df['review'].apply(extract_aspect)
    df['sentimen'] = df['predicted_sentiment'].apply(map_sentiment)

    # Filter hanya yang Positif dan Negatif
    df = df[df['sentimen'].isin(['Positif', 'Negatif'])]

    st.subheader("📊 Distribusi Sentimen per Aspek")
    color_map = {'Negatif': '#e74c3c', 'Positif': '#3498db'}
    aspek_order = ['Lagu', 'Harga', 'Tutorial', 'Login', 'Teknis']
    sentimen_order = ['Negatif', 'Positif']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df,
        x='aspect',
        hue='sentimen',
        order=aspek_order,
        hue_order=sentimen_order,
        palette=color_map,
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9)

    ax.set_title("Distribusi Sentimen per Aspek (ABSA) - Ukulele by Yousician", fontsize=14, weight='bold')
    ax.set_xlabel("Aspek")
    ax.set_ylabel("Jumlah Ulasan")
    ax.legend(title="Sentimen")
    st.pyplot(fig)

    # Unduh hasil
    st.subheader("📥 Unduh Hasil dengan Aspek dan Sentimen")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📤 Download CSV", data=csv, file_name="hasil_absa_yousician.csv", mime='text/csv')

    st.dataframe(df[['name', 'star_rating', 'date', 'review', 'aspect', 'sentimen']])

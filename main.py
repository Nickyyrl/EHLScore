from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
import plotly.express as px

# Configuration de la page pour une mise en page large
st.set_page_config(
    page_title="Application de scoring des leads",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Fonction pour calculer le score (inchangé)
def calculate_score(lead, idx, total, progress_bar):
    score = 0
    explanations = []
    progress_bar.progress((idx + 1) / total)

    # Critères de scoring
    if pd.notna(lead['clickedAt']):
        clicked_date = lead['clickedAt']
        if isinstance(clicked_date, pd.Timestamp):
            clicked_date = clicked_date.to_pydatetime()
        if (datetime.now() - clicked_date).days <= 7:
            score += 5
            explanations.append("5 points : Le lead a cliqué sur un lien dans les 7 derniers jours.")
        else:
            score += 3
            explanations.append("3 points : Le lead a cliqué sur un lien il y a plus de 7 jours.")

    if pd.notna(lead['sentStep']):
        try:
            step = int(lead['sentStep'])
            if step == 9:
                score += 5
                explanations.append("5 points : Le lead est à l'étape finale de la séquence.")
            elif step >= 7:
                score += 4
                explanations.append("4 points : Le lead est dans les étapes avancées.")
            elif step >= 5:
                score += 3
                explanations.append("3 points : Le lead est à une étape moyenne.")
            elif step >= 3:
                score += 2
                explanations.append("2 points : Le lead est dans les premières étapes.")
            else:
                score += 1
                explanations.append("1 point : Le lead est à l'étape initiale.")
        except ValueError:
            explanations.append("0 point : La valeur de l'étape n'est pas valide.")

    if pd.notna(lead['meetingBooked']):
        score += 5
        explanations.append("5 points : Une réunion a été réservée par le lead.")
    elif pd.notna(lead['interestedAt']):
        score += 3
        explanations.append("3 points : Le lead a marqué un intérêt explicite.")

    if lead['Statut de l_op_rateur'] == 'Régie à autonomie financière':
        score += 2
        explanations.append("2 points : Le lead a un opérateur actif ('Régie à autonomie financière').")

    if 'openedAt' in lead and pd.notna(lead['openedAt']):
        score += 3
        explanations.append("3 points : L'email envoyé a été ouvert par le lead.")

    score = min(score, 20)
    return score, explanations


# Fonction pour formater les numéros de téléphone
def format_phone_number(phone):
    phone = str(phone)
    if not phone.startswith('0'):
        phone = '0' + phone
    return ' '.join([phone[i:i + 2] for i in range(0, len(phone), 2)])


# Fonction pour formater les dates
def format_date(date_str):
    try:
        date_obj = pd.to_datetime(date_str)
        return date_obj.strftime('%d/%m/%Y')
    except:
        return 'Non précisé'


# Fonction pour préparer les données pour le clustering
def preprocess_data(df):
    features = ['Population municipale 2021', 'Statut de l_op_rateur', 'sentStep', 'Co_t annuel des fuites']
    data = df[features].copy()

    # Encodage des variables catégoriques
    if 'Statut de l_op_rateur' in data.columns:
        le = LabelEncoder()
        data['Statut de l_op_rateur'] = le.fit_transform(data['Statut de l_op_rateur'].astype(str))

    # Remplissage des valeurs manquantes
    data = data.fillna(0)

    # Normalisation des données
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    return normalized_data, data


# Déterminer le nombre optimal de clusters
def find_optimal_clusters(data, max_clusters=10):
    silhouettes = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        silhouettes.append(silhouette_score(data, kmeans.labels_))
    optimal_k = np.argmax(silhouettes) + 2  # Ajustement pour correspondre à la plage de k
    return optimal_k, silhouettes


# Appliquer le clustering
def cluster_leads(df):
    normalized_data, original_data = preprocess_data(df)

    # Trouver le nombre optimal de clusters
    optimal_k, silhouettes = find_optimal_clusters(normalized_data)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)

    # Ajouter les clusters au DataFrame
    df['Profil'] = clusters

    return df, optimal_k, silhouettes


# Générer des descriptions de clusters
def describe_clusters(df):
    cluster_summary = {}
    for cluster_id in sorted(df['Profil'].unique()):
        cluster_data = df[df['Profil'] == cluster_id]
        summary = {
            "Taille du cluster": len(cluster_data),
            "Population moyenne": cluster_data['Population municipale 2021'].mean(),
            "Statut le plus fréquent": cluster_data['Statut de l_op_rateur'].mode()[0],
            "Étape moyenne de séquence": cluster_data['sentStep'].mean(),
            "Coût moyen des fuites": cluster_data['Co_t annuel des fuites'].mean()
        }
        cluster_summary[cluster_id] = summary
    return cluster_summary


def perform_pca(df, features, clusters_column):
    # Normalisation des données avant PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Réalisation de l'ACP
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    # Création d'un DataFrame pour la visualisation
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df[clusters_column].values

    # Variance expliquée
    explained_variance = pca.explained_variance_ratio_

    return df_pca, explained_variance


# Fonction principale
def main():
    st.title("Application de scoring des leads")
    logo = "logo.png"
    st.markdown(
        f"""
        <div style='position: fixed; top: 10px; left: 10px;'>
            <img src='{logo}' width='50'>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)

        total_leads = len(df)
        progress_bar = st.progress(0)

        # Calcul des scores
        scores = []
        explanations_list = []
        for idx, lead in df.iterrows():
            score, explanations = calculate_score(lead, idx, total_leads, progress_bar)
            scores.append(score)
            explanations_list.append(explanations)

        df['score'] = scores
        df['score_explanations'] = explanations_list

        if 'phone' in df.columns:
            df['formatted_phone'] = df['phone'].apply(format_phone_number)

        # Affichage des résultats
        display_option = st.radio(
            "Choisissez l'option d'affichage :",
            ('Afficher les 10 meilleurs leads', 'Afficher tous les leads')
        )

        if display_option == 'Afficher les 10 meilleurs leads':
            leads_to_display = df.nlargest(10, 'score')
        else:
            leads_to_display = df

        st.subheader("Résultats des leads")
        st.dataframe(leads_to_display[['lastName', 'firstName', 'email', 'formatted_phone', 'score']])

        lead_selected = st.selectbox(
            "Sélectionnez un lead pour voir les détails",
            leads_to_display['lastName'] + ' ' + leads_to_display['firstName']
        )

        selected_lead = leads_to_display[
            leads_to_display['lastName'] + ' ' + leads_to_display['firstName'] == lead_selected].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Détails du Lead")
            st.write(f"Nom: {selected_lead['lastName']}")
            st.write(f"Prénom: {selected_lead['firstName']}")
            st.write(f"Email: {selected_lead['email']}")
            st.write(f"Téléphone: {selected_lead['formatted_phone']}")

            st.markdown(
                f"<h1 style='color: #009EE3; font-size: 50px; font-weight: bold;'>Score: {selected_lead['score']} / 20</h1>",
                unsafe_allow_html=True
            )
            st.write(f"Étape de la séquence: {selected_lead['sentStep']}")
            st.write(f"Réunion réservée: {selected_lead['meetingBooked']}")
            st.write(f"Email ouvert: {'Oui' if pd.notna(selected_lead['openedAt']) else 'Non'}")
            st.write(f"Date d'engagement: {format_date(selected_lead['interestedAt'])}")
            st.write(f"Date de clic: {format_date(selected_lead['clickedAt'])}")

            # Partie principale dans la fonction 'main()' modifiée
            st.subheader("📊 STATS BONUS : Analyse des Profils Types")
            df, optimal_k, silhouettes = cluster_leads(df)
            cluster_summary = describe_clusters(df)

            # Présentation du nombre optimal de profils
            st.markdown(f"""
            <div style="text-align: center; font-size: 24px; color: #009EE3; font-weight: bold;">
                🚀 Nombre optimal de profils détecté : <span style="font-size: 30px;">{optimal_k}</span>
            </div>
            """, unsafe_allow_html=True)

            # Présentation des clusters sous forme de colonnes
            cols = st.columns(int(optimal_k))  # Nombre de colonnes pour chaque profil
            for i, (cluster_id, summary) in enumerate(cluster_summary.items()):
                with cols[i]:
                    st.markdown(f"""
                <div style="background-color: black; padding: 20px; border-radius: 15px; text-align: left; width: 300px; margin: 10px;">
                    <h4 style="color: #007BFF; margin-bottom: 10px; font-size: 18px;">Profil {cluster_id}</h4>
                        <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
                            <li><b>Taille :</b> {summary['Taille du cluster']}</li>
                            <li><b>Population moyenne :</b> {summary['Population moyenne']:.2f}</li>
                            <li><b>Statut fréquent :</b> {summary['Statut le plus fréquent']}</li>
                            <li><b>Étape moyenne :</b> {summary['Étape moyenne de séquence']:.2f}</li>
                            <li><b>Coût moyen des fuites :</b> {summary['Coût moyen des fuites']:.2f} €</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            # Ajout de l'ACP et graphique interactif
            st.markdown("### 🌐 Visualisation des Profils Types par ACP")
            features_to_use = ['Population municipale 2021', 'sentStep', 'Co_t annuel des fuites']
            df_pca, explained_variance = perform_pca(df, features=features_to_use, clusters_column='Profil')

            # Affichage du graphique ACP
            fig_pca = px.scatter(
                df_pca,
                x='PC1', y='PC2',
                color='Cluster',
                title=f"Répartition des Profils Types (Variance expliquée : {sum(explained_variance) * 100:.2f}%)",
                labels={'PC1': 'Composante Principale 1', 'PC2': 'Composante Principale 2'},
                template='plotly_white'
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            # Tableau récapitulatif des clusters
            st.markdown("### 📝 Tableau Récapitulatif des Profils Types")
            summary_table = pd.DataFrame.from_dict(cluster_summary, orient='index')
            summary_table = summary_table.rename(columns={
                "Taille du cluster": "Taille",
                "Population moyenne": "Population Moyenne",
                "Statut le plus fréquent": "Statut Fréquent",
                "Étape moyenne de séquence": "Étape Moyenne",
                "Coût moyen des fuites": "Coût Moyen (€)"
            })
            st.dataframe(summary_table.style.format({"Population Moyenne": "{:.2f}", "Coût Moyen (€)": "{:.2f}"}))


if __name__ == "__main__":
    main()

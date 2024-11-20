import streamlit as st
import pandas as pd
from datetime import datetime

# Configuration de la page pour une mise en page large
st.set_page_config(
    page_title="Application de scoring des leads",
    layout="wide",  # Utilisation de toute la largeur de la page
    initial_sidebar_state="expanded"
)

# Calcul du score pour chaque lead
def calculate_score(lead, idx, total, progress_bar):
    score = 0
    progress_bar.progress((idx + 1) / total)

    # Critères de scoring
    if pd.notna(lead['clickedAt']):
        clicked_date = lead['clickedAt']
        if isinstance(clicked_date, pd.Timestamp):
            clicked_date = clicked_date.to_pydatetime()
        if (datetime.now() - clicked_date).days <= 7:
            score += 5
        else:
            score += 3

    if pd.notna(lead['sentStep']):
        try:
            step = int(lead['sentStep'])
            if step == 9:
                score += 5
            elif step >= 7:
                score += 4
            elif step >= 5:
                score += 3
            elif step >= 3:
                score += 2
            else:
                score += 1
        except ValueError:
            score += 0

    if pd.notna(lead['meetingBooked']):
        score += 5
    elif pd.notna(lead['interestedAt']):
        score += 3
    else:
        score += 0

    if lead['Statut de l_op_rateur'] == 'Actif':
        score += 2
    else:
        score += 0

    if 'openedAt' in lead and pd.notna(lead['openedAt']):
        score += 3
    else:
        score += 0

    score = min(score, 20)
    return score


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

        scores = []
        for idx, lead in df.iterrows():
            score = calculate_score(lead, idx, total_leads, progress_bar)
            scores.append(score)

        df['score'] = scores

        if 'phone' in df.columns:
            df['formatted_phone'] = df['phone'].apply(format_phone_number)

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

            st.write(f"Etape de la séquence: {selected_lead['sentStep']}")
            st.write(f"Réunion réservée: {selected_lead['meetingBooked']}")
            st.write(f"Email ouvert: {'Oui' if pd.notna(selected_lead['openedAt']) else 'Non'}")
            st.write(f"Date d'engagement: {format_date(selected_lead['interestedAt'])}")
            st.write(f"Date de clic: {format_date(selected_lead['clickedAt'])}")

        with col2:
            st.write(f"Statut de l'opérateur: {selected_lead['Statut de l_op_rateur']}")
            st.write(
                f"Coût annuel des fuites (€): {selected_lead['Co_t annuel des fuites'] if pd.notna(selected_lead['Co_t annuel des fuites']) else 'Non précisé'}")
            st.write(
                f"Population: {selected_lead['Population municipale 2021'] if pd.notna(selected_lead['Population municipale 2021']) else 'Non précisé'}")


if __name__ == "__main__":
    main()

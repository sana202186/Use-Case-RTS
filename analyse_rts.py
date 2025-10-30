metrics = pd.read_csv("sample_data/Mesures.csv", encoding="latin-1", sep=";")
tags = pd.read_csv("sample_data/Tags.csv", encoding="latin-1", sep=";")
print("Fichiers chargés")
print("Tags :", tags.shape, " | Metrics :", metrics.shape)
# Suppression des espaces dans les noms de colonnes
tags.columns = tags.columns.str.strip()
metrics.columns = metrics.columns.str.strip()

# Suppression des espaces superflus dans les cellules
for df in [tags, metrics]:
    df.replace(r'^\s+|\s+$', '', regex=True, inplace=True)

# Suppression des lignes entièrement vides
tags.dropna(how='all', inplace=True)
metrics.dropna(how='all', inplace=True)

# Harmonisation des identifiants 
tags['Segment ID'] = tags['Segment ID'].astype(str).str.strip()
metrics['Segment ID'] = metrics['Segment ID'].astype(str).str.strip()

#  Nettoyage des valeurs numériques et textuelles 
# Conversion du taux de nouvelles visites en float
if 'New Visit Rate %' in metrics.columns:
    metrics['New Visit Rate %'] = (
        metrics['New Visit Rate %']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    metrics['New Visit Rate %'] = pd.to_numeric(metrics['New Visit Rate %'], errors='coerce')

# Conversion du temps moyen de lecture en secondes
def time_to_seconds(t):
    try:
        h, m, s = map(int, str(t).split(':'))
        return h*3600 + m*60 + s
    except:
        return np.nan

if 'Avg Play Duration' in metrics.columns:
    metrics['Avg Play Duration (s)'] = metrics['Avg Play Duration'].apply(time_to_seconds)
if 'Total Play Duration' in metrics.columns:
    metrics['Total Play Duration (s)'] = df['Total Play Duration'].apply(time_to_seconds)

# Suppression des doublons
tags.drop_duplicates(subset="Segment ID", inplace=True)
metrics.drop_duplicates(subset="Segment ID", inplace=True)

print("Fichiers nettoyés")
print("Tags :", tags.shape, " | Metrics :", metrics.shape)

# === Vérifications avant fusion ===
common_ids = set(tags['Segment ID']) & set(metrics['Segment ID'])
print(f"\n?? Vérification des identifiants :")
print(f"- IDs communs : {len(common_ids)}")
print(f"- IDs uniquement dans Mesures : {len(set(metrics['Segment ID']) - common_ids)}")
print(f"- IDs uniquement dans Tags : {len(set(tags['Segment ID']) - common_ids)}")

# ===  Fusion ===
merged = pd.merge(
    metrics,
    tags[['Segment ID', 'Assigned Tags']],
    on='Segment ID',
    how='left'
)
print("Dimensions finales :", merged.shape)
#  Sauvegarde du fichier propre ===
output_path = "sample_data/Metrics_Tags_Clean.csv"
merged.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f" Fichier nettoyé et fusionné enregistré sous : {output_path}")

# =Aperçu des données ===
#display(merged.head(10))

###### For LDA model training
df = merged
# Fonction de nettoyage
def clean_tag(text):
    text = text.lower()
    text = re.sub(r'media_radio|rts_info|la-1ere|media_tv|couleur3|podcasts-originaux', '', text)
    text = text.replace('_', ' ').replace(':', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned'] = df['Assigned Tags'].apply(clean_tag)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words=None)
X = vectorizer.fit_transform(df['Cleaned'])
from sklearn.decomposition import LatentDirichletAllocation

# Exemple : on veut trouver 10 thèmes
lda = LatentDirichletAllocation(n_components=20, random_state=42)
lda.fit(X)
words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(f"Thème {i+1} :", ", ".join(top_words))

##to classfiy tags
import re

# 1. Remplir les NaN
merged['Assigned Tags'] = merged['Assigned Tags'].fillna('')

# 2. Split + nettoyage
def clean_tags(tag_string):
    # Séparer par ":" ou ","
    tags = re.split(':|,', tag_string)
    cleaned = []
    for t in tags:
        t = t.strip().lower()
        # Supprimer les préfixes/éléments récurrents
        t = re.sub(r'\b(media_radio|la-1ere|rts_info|podcasts|espace-2|consommation|media|tv|couleur3|valais)\b', '', t)
        t = t.strip('_-: ')
        if t:  # ne garder que les tags non vides
            cleaned.append(t)
    return cleaned

merged['cleaned_tags'] = merged['Assigned Tags'].apply(clean_tags)

# 3. Fonction pour assigner le thème
theme_keywords = {
    'Info': ['info', 'reportages', 'news', 'economie' ,'monde'],
    'Sport': ['sport', 'match', 'football', 'basket', 'tennis', 'rugby', 'athlétisme'],
    'Musique': ['musique', 'concert', 'chanson', 'album', 'pop', 'rock', 'classique'],
    'Société': ['societe', 'entretiens', 'social', 'documentaire social', 'culture'],
    'Humour': ['humour', 'comedy', 'blague', 'sketch', 'stand-up']
}


def assign_theme(tags_list):
    for tag in tags_list:
        for theme, keywords in theme_keywords.items():
            if any(kw in tag for kw in keywords):
                return theme
    return 'Autre'

merged['Theme_Final'] = merged['cleaned_tags'].apply(assign_theme)

# 4. Vérifier le résultat
print(merged['Theme_Final'].value_counts())

###Définition des métriques

theme_metrics = merged.groupby('Theme_Final').agg({
    'Segment ID': 'count',                  # nombre de segments produits
    'Media Views': 'sum',                   # total des vues
    'Visitors': 'sum',                      # nombre de visiteurs uniques
    'Returning Visits': 'sum',              # visites de retour
    'Bounces': 'sum',                        # rebonds
    'Total Play Duration (s)': 'sum',       # engagement total
    'Avg Play Duration (s)': 'mean',         # durée moyenne de visionnage
    'New Visit Rate %': 'mean' # Add mean of New Visit Rate %
}).rename(columns={'Segment ID':'Num_Segments'})

# Calculer les scores Acquisition, Retention, Engagement ---
theme_metrics['Acquisition_Score'] = theme_metrics['Visitors'] * theme_metrics['New Visit Rate %'] / 100
theme_metrics['Retention_Score'] = theme_metrics['Returning Visits']
theme_metrics['Engagement_Score'] = theme_metrics['Total Play Duration (s)']

# Normalisation pour comparer facilement
for col in ['Acquisition_Score','Retention_Score','Engagement_Score']:
    theme_metrics[col+'_norm'] = theme_metrics[col] / theme_metrics[col].max()
# Priorisation des thèmes pour production ---
theme_metrics['Priority_Score'] = 0.6*theme_metrics['Acquisition_Score_norm'] + 0.4*theme_metrics['Retention_Score_norm']
theme_metrics = theme_metrics.sort_values('Priority_Score', ascending=False)

# Création d'une série temporelle synthétique
dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
records = []

for theme, row in theme_metrics.iterrows():
    base = row['Visitors']
    growth = np.random.uniform(0.01, 0.05)  # croissance mensuelle simulée
    for i, d in enumerate(dates):
        visitors = base * ((1 + growth) ** i) / 12
        records.append({
            "Date": d,
            "Theme_Final": theme,
            "Visitors": visitors
        })

theme_time = pd.DataFrame(records)
print("Série temporelle générée :", theme_time.shape)
print(theme_time.head())

# Courbe d’évolution des visiteurs ---
plt.figure(figsize=(12,6))
sns.lineplot(data=theme_time, x='Date', y='Visitors', hue='Theme_Final', marker="o")
plt.title("Évolution temporelle des visiteurs par thème")
plt.ylabel("Visiteurs")
plt.xlabel("Date")
plt.show()

#  Version interactive 
fig = px.line(theme_time, x='Date', y='Visitors', color='Theme_Final',
              title='Évolution temporelle interactive des visiteurs par thème')
fig.show()



import streamlit as st  
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

if 'reviews_df' in st.session_state:
    df = st.session_state.reviews_df.copy()

    # Extraire les infos sous forme de DataFrame
    df_info = pd.DataFrame({
        "Colonnes": df.columns,
        "Type": df.dtypes.astype(str),
        "Valeurs Non Nul": df.count(),
        "Valeurs Manquantes": df.isna().sum(),
        "% Manquantes": (df.isna().sum() / len(df) * 100).round(2)
    }).reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["Colonnes", "Graphiques", "Tests statistiques"])

    with tab1:

        st.write("📊 Informations du DataFrame :")
        st.dataframe(df_info)
        
        st.write("Informations sur les colonnes du DataFrame")
        for column in df.columns:
            with st.expander(f"Colonne: {column}"):
                st.write(f"**{column}**")
                st.write(f"Type: {df[column].dtype}")
                st.write(f"Nombre de valeurs uniques: {df[column].nunique()}")
                st.write(f"Valeurs manquantes: {df[column].isnull().sum()}")
                if df[column].dtype == 'object':
                    st.write("Valeurs les plus fréquentes:")
                    st.dataframe(df[column].value_counts().head().reset_index().rename(columns={'index': column, column: 'Count'}))
                else:
                    st.write(f"Statistiques descriptives:")
                    st.dataframe(df[column].describe())
                st.write("---")

    with tab2:
        with st.expander("Histogramme de la répartition des notes (vérifiés vs non vérifiés)"):
            if df['Vérifié'].dtype == bool:
                verified = df[df['Vérifié'] == True]
                non_verified = df[df['Vérifié'] == False]
            else:
                verified = df[df['Vérifié'] == "Vérifié"]
                non_verified = df[df['Vérifié'] == "Non vérifié"]

            hist_values_verified = np.histogram(verified['Note'], bins=5, range=(1, 6))[0]
            hist_values_non_verified = np.histogram(non_verified['Note'], bins=5, range=(1, 6))[0]

            bar_width = 0.35
            r1 = np.arange(1, 6)
            r2 = [x + bar_width for x in r1]

            plt.figure()
            plt.bar(r1, hist_values_verified, color='blue', width=bar_width, edgecolor='grey', label='Vérifiés')
            plt.bar(r2, hist_values_non_verified, color='red', width=bar_width, edgecolor='grey', label='Non vérifiés')

            plt.xlabel('Notes')
            plt.ylabel('Nombre d\'avis')
            plt.xticks([r + bar_width/2 for r in range(1, 6)], [1, 2, 3, 4, 5])
            plt.legend()
            st.pyplot(plt)
            st.write("On peut constater que l'énorme majorité des avis sont positifs, avec une note de 5/5. Ce qui est étonnant, car on pourrait s'attendre à ce que les gens insatisfaits viennent plus souvent s'exprimer.")
            
            st.write("On pourrait penser que les mauvais avis peuvent faire l'objet de mensonges ou de campagne de dénigrement de personnes non vérifiées, vérifions.")
            st.write("Distribution des notes (vérifié / non vérifié / total) :")
            note_distribution = df.groupby(['Note', 'Vérifié']).size().unstack(fill_value=0)
            note_distribution = note_distribution.div(note_distribution.sum(axis=1), axis=0) * 100
            note_distribution['Total'] = note_distribution.sum(axis=1)
            note_distribution['Total'] = df['Note'].value_counts(normalize=True) * 100
            note_distribution = note_distribution.applymap(lambda x: f"{x:.2f}%")
            st.dataframe(note_distribution)

            verified_avg = verified['Note'].mean()
            non_verified_avg = non_verified['Note'].mean()

            st.write(f"Note moyenne des avis vérifiés: {verified_avg:.2f}")
            st.write(f"Note moyenne des avis non vérifiés: {non_verified_avg:.2f}")

            rate_1_verified = len(verified[verified['Note'] == 1]) / len(df[df['Note'] == 1])
            rate_5_verified = len(verified[verified['Note'] == 5]) / len(df[df['Note'] == 5])
            st.write("On a finalement une note moyenne quasiment identique, on constate cependant que les avis une étoile ont un taux plus important de non vérifié par rapport aux 5 étoiles, mais plus globalement que les avis non vérifiés on tendance à être plus extremes.")
        with st.expander("Scatterplot entre la longueur des avis et la note"):
            df['Avis_length'] = df['Avis'].fillna("").apply(len)
            plt.figure()
            plt.scatter(df['Note'], df['Avis_length'])
            plt.plot(np.unique(df['Note']), np.poly1d(np.polyfit(df['Note'], df['Avis_length'], 1))(np.unique(df['Note'])), color='red')
            plt.xlabel('Longueur des avis')
            plt.ylabel('Note')
            st.pyplot(plt)
            st.write("On peut constater une légère tendance à ce que les avis plus longs aient tendance à donner des notes moins élevées. Ce qui semble logique vu que les gens insatisfaits ont plus de choses à dire et de motivation à détailler leur avis.")
        
        with st.expander("Taux de réponse aux avis et note moyenne par entreprise"):
            response_rate = df['Réponse'].apply(lambda x: not pd.isnull(x) and x != "Pas de réponse").groupby(df['Entreprise']).mean()
            average_rating = df.groupby('Entreprise')['Note'].mean()

            fig, ax1 = plt.subplots()

            ax1.set_xlabel('Note moyenne')
            ax1.set_ylabel('Taux de réponse')
            ax1.set_ylim(0, 1.0)  # Limiter l'axe y à 1.0 maximum
            ax1.scatter(average_rating.values, response_rate.values, color='blue', marker='o')
            sns.regplot(x=average_rating.values, y=response_rate.values, ax=ax1, color='red', scatter=False)
            fig.tight_layout()
            st.pyplot(fig)

            st.text("On peut constater que les entreprises ayant une note moyenne plus élevée ont tendance à répondre moins souvent aux avis. Ce qui tendrait à indiquer que les entreprises ayant une meilleure réputation n'ont pas besoin de répondre à tous les avis pour maintenir leur image, et qu'à l'inverse, les entreprises avec de moins bonnes notes sont soucieuses d'améliorer leur image et de répondre à leurs avis.")

        with st.expander("Évolution dans le temps"):
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            fig, ax = plt.subplots(3, 1, figsize=(10, 15))

            df['Note'].resample('M').mean().plot(ax=ax[0], color='blue')
            ax[0].set_title('Évolution des notes moyennes par mois')
            ax[0].set_xlabel('Date')
            ax[0].set_ylabel('Note moyenne')

            df['Avis_length'].resample('M').mean().plot(ax=ax[1], color='green')
            ax[1].set_title('Évolution de la longueur moyenne des avis par mois')
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Longueur moyenne des avis')

            df['Avis'].resample('M').count().plot(ax=ax[2], color='purple')
            ax[2].set_title('Évolution du nombre d\'avis par mois')
            ax[2].set_xlabel('Date')
            ax[2].set_ylabel('Nombre d\'avis')

            plt.tight_layout()
            st.pyplot(fig)

            st.text("Les deux premiers graphiques ne montrent pas de tendance intéressante, par contre le dernier montre deux choses : une apparente saisonnalité dans le dépôt d'avis, et surtout une apparente croissance du nombre d'avis.")
            st.text("On constate plusieurs trous dans les courbes, pas assez de data sur ces périodes ? pertes de données ?")
        with st.expander("Analyse de la saisonnalité du nombre d'avis"):
            df['Month'] = df.index.month
            monthly_reviews = df['Avis'].groupby(df['Month']).count()

            plt.figure(figsize=(10, 6))
            sns.barplot(x=monthly_reviews.index, y=monthly_reviews.values, palette='viridis')
            plt.xlabel('Mois')
            plt.ylabel('Nombre d\'avis')
            plt.title('Nombre d\'avis par mois')
            st.pyplot(plt)

            st.write("On peut observer un nombre d'avis plus important en début d'année, date d'entrée / sortie de la plupart des promos ? On constate aussi un creux en août explicable par les congés et la baisse d'activité sur les formations ainsi qu'en avril-mai, où on peut penser que la plupart des formations sont en cours et donc que les gens n'en sont pas encore au moment de s'exprimer.")
        with st.expander("Analyse de la tendance en éliminant la saisonnalité"):

            # Décomposer la série temporelle
            decomposition = seasonal_decompose(df['Avis'].resample('M').count(), model='additive')
            trend = decomposition.trend

            plt.figure(figsize=(10, 6))
            plt.plot(trend, label='Tendance', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Nombre d\'avis')
            plt.title('Tendance du nombre d\'avis par mois (sans saisonnalité)')
            plt.legend()
            st.pyplot(plt)

            st.write("La tendance montre effectivement une croissance, rendant d'autant plus intéressante notre solution puisque plus on a d'avis, plus cela nécessite de travail de réponse.")
            #TODO: Analyse du nombre d'avis par entreprise
        with st.expander("Nuage de mots des mots les plus fréquents par note"):
            nltk.download('stopwords')
            stop_words = set(stopwords.words('french'))
            additional_stop_words = ["formation", "j'ai", "c'est", "être","a","n'est","comme","d'un"]
            stop_words.update(additional_stop_words)
            for note in range(1, 6):
                st.subheader(f"Note {note}")
                text = " ".join(review for review in df[df['Note'] == note]['Avis'].dropna())
                text = " ".join(word for word in text.split() if word.lower() not in stop_words)

                wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", min_word_length=4).generate(text)

                plt.figure()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
            st.write("Difficile de tirer des conclusions sans plus de nettoyage et d'affinage, quelques mots importants ressortent cependant : Site, plateforme, formateur, accompagnement, suivi. \n Ces thèmes sont donc probablement importants dans le cadre d'une formation en ligne. ")

    with tab3:
        st.write("Tests statistiques à venir")

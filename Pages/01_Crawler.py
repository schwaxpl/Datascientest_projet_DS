
import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time


def get_company_urls(category_url):
    company_urls = []
    page = 1
    status = st.empty()
    while True:
        response = requests.get(f"{category_url}?page={page}")
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, 'html.parser')
        company_cards = soup.find_all('a', class_='link_internal__Eam_b link_wrapper__ahpyq styles_linkWrapper___KiUr')
        if not company_cards:
            break
        for card in company_cards:
            href = card.get('href')
            if href and href.startswith('/review/'):
                company_urls.append(href.split('/review/')[1])
        page += 1
        time.sleep(0.3)
        status.text(f"Récupération des entreprises page {page}...")
    return company_urls

def get_reviews(company_url):
    response = requests.get(company_url)
    status = st.empty()
    review_cards = []
    #review_cards = soup.find_all('article', class_=lambda x: x and x.startswith('styles_reviewCard'))
    page = 1
    while True:
        print(page)
        response = requests.get(f"{company_url}?page={page}")
        if response.status_code != 200:
            break
        print(response.status_code)
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews_container = soup.find('section', class_=lambda x: x and x.startswith('styles_reviewListContainer'))
        if reviews_container:
            new_review_cards = reviews_container.find_all('article', class_=lambda x: x and x.startswith('styles_reviewCard')) 
            print(new_review_cards[0])
        else:
            new_review_cards = []
        #new_review_cards = soup.find_all('article', class_=lambda x: x and x.startswith('styles_reviewCard'))
        if not new_review_cards:
            break
        review_cards.extend(new_review_cards)
        page += 1
        time.sleep(0.3) 
        status.text(f"Récupération des avis page {page}...")
    reviews = []
    for card in review_cards:
        print(type(card))
        rating_element = card.find('div', class_=lambda x: x and x.startswith('star-rating'))
        if rating_element:
            rating_img = rating_element.find('img')
            if rating_img:
                rating = rating_img['src'].split('/')[-1].split('-')[-1].split('.')[0]
            else:
                rating = "Pas de note"
        else:
            rating = "Pas de note"
        review_text_element = card.find('p', class_=lambda x: x and x.startswith('typography_body-l'))
        review_text = review_text_element.text.strip() if review_text_element else "Pas de texte d'avis"
        title_element = card.find('h2', class_=lambda x: x and x.startswith('typography_heading'))
        title_text = title_element.text.strip() if title_element else "Pas de titre"
        author_element = card.find('span', class_=lambda x: x and x.startswith('typography_heading'))
        author_name = author_element.text.strip() if author_element else "Auteur inconnu"
        reply_element = card.find('div', class_=lambda x: x and x.startswith('styles_replyHeader'))
        if reply_element:
            reply_text_element = reply_element.find_next_sibling('p')
            reply_text = reply_text_element.text.strip() if reply_text_element else "Pas de réponse"
        else:
            reply_text = "Pas de réponse"
        
        reviews.append({"Titre": title_text, "Note": rating, "Avis": review_text, "Auteur": author_name, "Réponse": reply_text})

    
    return reviews

tab1, tab2 = st.tabs(["Par secteur d'activité", "Par entreprise"])

with tab1:
    st.title("Par secteur d'activité")
    sector = st.text_input("Secteur d'activité")  
    if st.button("Obtenir les entreprises"):
        if sector:
            url = f"https://fr.trustpilot.com/categories/{sector}"
            company_urls = get_company_urls(url)
            st.session_state.company_urls = company_urls
            st.session_state.sector = sector
            st.write(company_urls)
        else:
            st.write("Veuillez entrer un secteur d'activité.")


    if 'company_urls' in st.session_state and 'sector' in st.session_state:
        st.write(f"Entreprises du secteur {st.session_state.sector} en mémoire.")
        st.write(f"Nombre d'entreprises: {len(st.session_state.company_urls)}")
        company_urls = st.session_state.company_urls
        sector = st.session_state.sector
        cols = st.columns(2)
        with cols[0]:
            start_from = st.text_input("À partir de l'entreprise n°", value="1")
            obtenir_avis =  st.button("Obtenir les avis pour ces entreprises")
        with cols[1]:
            until = st.text_input("Jusqu'à l'entreprise n°", value=str(len(company_urls)))
            ecraser_avis =  st.checkbox("écraser les avis précédents")
        if obtenir_avis:
            if ecraser_avis:
                st.session_state.reviews_df = pd.DataFrame()
            stop_button = st.button("Interrompre et sauvegarder")
            entreprise = st.empty()
            start_index = int(start_from) - 1
            end_index = int(until)

            for company in company_urls[start_index:end_index]:
                url = f"https://fr.trustpilot.com/review/{company}"
                entreprise.text(f"Récupération des avis pour {company} (Entreprise {company_urls.index(company) + 1}/{len(company_urls)})...")
                st.session_state.last_company = company
                reviews = get_reviews(url)
                for review in reviews:
                    review['Entreprise'] = company
                    review['Secteur'] = sector
                
                company_reviews_df = pd.DataFrame(reviews)
                if 'reviews_df' not in st.session_state:
                    st.session_state.reviews_df = company_reviews_df
                else:
                    st.session_state.reviews_df = pd.concat([st.session_state.reviews_df, company_reviews_df], ignore_index=True) 
    if 'last_company' in st.session_state:
        st.write(f"Dernière entreprise traitée: {st.session_state.last_company}")
        last_company_index = company_urls.index(st.session_state.last_company) + 1
        st.write(f"Index de la dernière entreprise traitée: {last_company_index}")
        Resume = st.button("Reprendre le traitement")

with tab2:
    st.title("Par entreprise")
    company_name = st.text_input("Nom de l'entreprise")
    if st.button("Obtenir les avis"):
        if company_name:
            url = f"https://fr.trustpilot.com/review/{company_name}"
            reviews = get_reviews(url)
            if 'reviews_df' not in st.session_state:
                st.session_state.reviews_df = pd.DataFrame()
            else:
            # Remove existing reviews for the company if they exist
                if 'Entreprise' in st.session_state.reviews_df.columns:
                    st.session_state.reviews_df = st.session_state.reviews_df[st.session_state.reviews_df['Entreprise'] != company_name]

            # Add new reviews with the company name
            new_reviews_df = pd.DataFrame(reviews)
            new_reviews_df['Entreprise'] = company_name
            st.write(new_reviews_df)
            st.session_state.reviews_df = pd.concat([st.session_state.reviews_df, new_reviews_df], ignore_index=True)


        else:
            st.write("Veuillez entrer un nom d'entreprise.")

if st.session_state.reviews_df.empty:
    st.write("Aucun avis à afficher pour le moment.")
else:
    st.text("Les 1000 premiers avis:")
    st.dataframe(st.session_state.reviews_df.head(1000))
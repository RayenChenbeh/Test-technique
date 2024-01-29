import pandas as pd
import numpy as np
import json

def produce_brut(assureur):
    with open('data/ocr/{}.json'.format(assureur), 'r',encoding="utf8") as f:
        data = json.load(f)
    pages_content=data["responses"]
    num_page=0
    df=[]
    for page in pages_content:
        num_page+=1
        if "fullTextAnnotation" not in page:
            continue
        p=page["fullTextAnnotation"]["pages"]
        for e in p:
            blocks=e["blocks"]
            page_features=[]
            for block in blocks:
                for para in block["paragraphs"]:
                    # collect text
                    text = ""
                    for word in para["words"]:
                        #print("-----")
                        #print(word)
                        for symbol in word["symbols"]:
                            if symbol["confidence"]>=0.8:
                                text += symbol["text"]
                        text+=" "
                    # extract bounding box features
                    x_list = []
                    y_list = []
                    for v in para["boundingBox"]["normalizedVertices"]:
                        x_list.append(v["x"])
                        y_list.append(v["y"])
                    f = {}
                    f["num_page"]=num_page
                    f["text"] = text
                    f["width"] = max(x_list) - min(x_list)
                    f["height"] = max(y_list) - min(y_list)
                    f["area"] = f["width"] * f["height"]
                    f["chars"] = len(text)

                    f['digit_count'] = sum(c.isdigit() for c in text)

                    f["char_size"] = f["area"] / f["chars"] if f["chars"] > 0 else 0
                    f["pos_x"] = (f["width"] / 2.0) + min(x_list)
                    f["pos_y"] = (f["height"] / 2.0) + min(y_list)
                    f["aspect"] = f["width"] / f["height"] if f["height"] > 0 else 0
                    f["layout"] = "h" if f["aspect"] > 1 else "v"
                    f["x0"]=x_list[0]
                    f["x1"]=x_list[1]
                    f["y0"]=y_list[0]
                    f["y1"]=y_list[1]
                    page_features.append(f)
            df=df+page_features   
    df=pd.DataFrame(df)
    df["assureur"]=assureur
    return df


def classify_text(row):
    # These thresholds would need to be refined after inspecting the data
    title_threshold_char_size = 0.000075
    paragraph_min_chars = 60
    useless_pos_y_threshold_upper = 0.9
    useless_pos_y_threshold_lower = 0.06
    max_witdh=0.32
    max_chars=30
    
    if (row['pos_y'] < useless_pos_y_threshold_upper and row['pos_y'] > useless_pos_y_threshold_lower) and row['char_size'] > title_threshold_char_size and row['chars'] < paragraph_min_chars and row['digit_count'] in [0,1,2,3]:
        return 'Title'
    elif (row['pos_y'] < useless_pos_y_threshold_upper and row['pos_y'] > useless_pos_y_threshold_lower) and row['width'] > max_witdh and row['chars'] > row['digit_count']and row['chars'] > max_chars :
        return 'Paragraph'
    else:
        return 'Useless'
    
    



df_allianz = produce_brut("allianz")
df_allianz.to_excel("allianz_data.xlsx", index=False)



# Apply the classification function to each row
df_allianz['Label'] = df_allianz.apply(classify_text, axis=1)
df_allianz.to_excel("allianz_data.xlsx", index=False)

# Affichage des premières lignes du DataFrame pour vérification
print(df_allianz.head())



import fitz  # Importez la bibliothèque PyMuPDF

# Assurez-vous que df_allianz est le DataFrame que vous avez obtenu après classification

def draw_rectangles_in_pdf(pdf_path, df):
    # Ouvrez le document PDF
    doc = fitz.open(pdf_path)

    # Définir les couleurs pour chaque type de contenu
    # En PyMuPDF, les couleurs sont définies par des tuples (r, g, b) avec des valeurs de 0 à 1
    colors = {'Title': (0, 0, 1),  # Bleu
              'Paragraph': (0, 0, 0),  # Noir
              'Useless': (1, 0, 0)}  # Rouge

    # Parcourir chaque ligne du DataFrame et dessiner un rectangle pour chaque bloc
    for index, row in df.iterrows():
        page = doc[row['num_page'] - 1]

        # Si les coordonnées sont normalisées (valeur entre 0 et 1), 
        # convertissez-les en coordonnées absolues
        x0 = row['x0'] * page.rect.width
        y0 = row['y0'] * page.rect.height
        x1 = row['x1'] * page.rect.width
        y1 = row['y1'] * page.rect.height


        # Créez un rectangle avec les coordonnées absolues
        rect = fitz.Rect(x0, y0, x1, y1)
        # Dessinez le rectangle avec un bord de couleur et remplissez partiellement transparent
        page.draw_rect(rect, color=colors[row['Label']], width=3.5, fill=colors[row['Label']], fill_opacity=0.3)

    # Enregistrez le PDF annoté
    output_pdf_path = "annotated_allianz.pdf"
    doc.save(output_pdf_path)
    doc.close()
    return output_pdf_path

# Le chemin vers votre fichier PDF
pdf_path = "data/pdfs/Allianz.pdf"  # Remplacez par le chemin correct
# Appliquez la fonction de dessin au PDF
annotated_pdf = draw_rectangles_in_pdf(pdf_path, df_allianz)

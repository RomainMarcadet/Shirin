from __future__ import annotations

import requests
import gradio as gr


API_BASE_URL = "http://localhost:8000"


def call_respond_endpoint(text: str, tone: str) -> tuple[str, str]:
    """
    Appelle l'endpoint /respond de l'API FastAPI et retourne
    (sentiment_label, texte_de_reponse_ou_message).
    """
    if not text.strip():
        return "aucun", "Veuillez saisir un avis client."

    payload = {
        "text": text,
        "tone": tone,
        "force": False,  # ne génère une réponse que si l'avis est négatif
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/respond", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return "erreur", f"Erreur lors de l'appel à l'API: {exc}"

    sentiment_label = data.get("sentiment_label", "inconnu")
    generated = data.get("generated", False)
    response_text = data.get("response_text")

    if not generated or not response_text:
        return sentiment_label, "Avis non négatif ou pas de réponse générée."

    return sentiment_label, response_text


def main() -> None:
    """
    Lance une interface Gradio pour tester :
    - la prédiction de sentiment,
    - la génération de réponse Mistral (polite ou sarcasm).
    """
    iface = gr.Interface(
        fn=call_respond_endpoint,
        inputs=[
            gr.Textbox(
                lines=5,
                label="Avis client",
                placeholder="Saisissez ici un avis client en français...",
            ),
            gr.Radio(
                choices=["polite", "sarcasm"],
                value="polite",
                label="Ton de la réponse",
            ),
        ],
        outputs=[
            gr.Textbox(label="Sentiment prédit (modèle TF-IDF)"),
            gr.Textbox(label="Réponse Mistral", lines=20),
        ],
        title="SAFE SAV - Classification & Réponse aux avis",
        description=(
            "Interface de test pour le pipeline :\n"
            "- classification des avis (négatif / neutre / positif)\n"
            "- génération de réponse personnalisée avec Mistral\n"
            "La réponse n'est générée que si l'avis est prédit négatif."
        ),
    )

    iface.launch(share=True)


if __name__ == "__main__":
    main()


import re

def is_loan_query(query: str) -> bool:
    """
    Détermine si la requête concerne un crédit ou un financement personnel.
    """
    keywords = ["crédit", "prêt", "personnel", "autofinancement", "emprunt", "salaire", "mensualité"]
    return any(word in query.lower() for word in keywords)

def parse_loan_params(query: str):
    """
    Extrait les paramètres de crédit depuis la requête :
    - autofinancement (apport personnel)
    - salaire mensuel
    - durée souhaitée
    """
    query = query.lower()
    num_pattern = r"(\d{3,6})\s*(dt|dinar[s]?)?"

    auto_match = re.search(r"autofinancement[=:]?\s*" + num_pattern, query)
    autofinancement = int(auto_match.group(1)) if auto_match else 0

    salaire_match = re.search(r"salaire[=:]?\s*" + num_pattern, query)
    salaire = int(salaire_match.group(1)) if salaire_match else None

    mois_match = re.search(r"(\d+)\s*mois", query)
    ans_match = re.search(r"(\d+)\s*(ans|an)", query)

    if mois_match:
        duration_months = int(mois_match.group(1))
    elif ans_match:
        duration_months = int(ans_match.group(1)) * 12
    else:
        duration_months = 60  # par défaut, 5 ans

    print(f"[🔍 DEBUG] Autofinancement: {autofinancement} DT")
    print(f"[🔍 DEBUG] Salaire: {salaire} DT")
    print(f"[🔍 DEBUG] Durée: {duration_months} mois")

    return {
        "autofinancement": autofinancement,
        "salaire": salaire,
        "duration_months": duration_months
    }

def simulate_loan(params: dict) -> str:
    """
    Simule un crédit personnel basé sur les revenus et l'apport de l'utilisateur.
    """
    salaire = params.get("salaire")
    autofinancement = params.get("autofinancement", 0)
    months = params.get("duration_months", 60)

    if salaire is None:
        return "❗ Veuillez indiquer votre salaire mensuel pour simuler un crédit."

    max_monthly_payment = salaire * 0.4
    interest_rate = 0.07
    monthly_rate = interest_rate / 12

    try:
        loan_amount = max_monthly_payment * ((1 - (1 + monthly_rate) ** -months) / monthly_rate)
        monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -months)
        total_amount = loan_amount + autofinancement

        response = (
            f"📅 Durée demandée : {months} mois\n"
            f"✅ Avec un salaire de {salaire} DT/mois et un autofinancement de {autofinancement} DT,\n"
            f"vous pouvez obtenir un crédit d’environ **{int(loan_amount):,} DT**.\n"
            f"💸 Mensualité estimée : {int(monthly_payment):,} DT/mois (max autorisé : {int(max_monthly_payment):,} DT/mois)\n"
            f"💰 Montant total disponible : environ **{int(total_amount):,} DT**."
        )

        if monthly_payment > max_monthly_payment:
            response += "\n⚠️ Attention : La mensualité dépasse 40% de votre salaire. Vous pourriez ne pas être éligible."

        return response

    except Exception as e:
        return f"❌ Erreur lors du calcul du crédit : {e}"

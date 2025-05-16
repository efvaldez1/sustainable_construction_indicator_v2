'''pip install streamlit PyPDF2 pandas altair sentence-transformers nltk scikit-learn'''

import streamlit as st
import PyPDF2
import re # Kept for existing structure, but new matching is NLP based
import pandas as pd
import altair as alt
from typing import List, Dict, Tuple

# --- NLP and Similarity Libraries ---
from sentence_transformers import SentenceTransformer
import nltk

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




# # --- Download NLTK resources (if not already downloaded) ---
# import urllib.error # exception error if there is problem with download
# try:
#     nltk.download('punkt')
#     # nltk.data.find('tokenizers/punkt')
#     # nltk.download('punkt_tab')
#     # nltk.download('wordnet')
#     # nltk.download('omw-1.4')
# # except nltk.downloader.DownloadError:
# #     nltk.download('punkt', quiet=True)
# except urllib.error.URLError:
#     print("Failed to download due to connectivity issue.")
# except LookupError:
#     print("NLTK resource not found.")

# --- Load Sentence Transformer Model (this will download the model on first run) ---
# Using a lightweight and effective model
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}. Ensure you have an internet connection for the first run.")
    st.stop()

# --- Similarity Threshold ---
SIMILARITY_THRESHOLD = 0.70 # 70% similarity

# --- Initial Keyword Matrix (Hardcoded from Concept Note) ---
initial_indicator_keywords = {
    "Building Energy Efficiency": ["HVAC efficiency", "U-value", "building envelope", "building envelope efficiency", "energy demand", "energy performance", "energy simulation", "natural ventilation", "passive solar", "thermal insulation"],
    "Energy Consumption": ["energy-efficient equipment", "fuel-efficient vehicles", "energy optimization", "low-energy site operations", "reduced generator use", "hybrid construction machinery", "site energy plan"],
    "Thermal Performance": ["thermal envelope", "insulation", "U-value", "heat loss"],
    "Fuel Type for Equipment": ["Biodiesel", "alternative fuel", "low sulfur diesel", "renewable diesel", "clean fuel specification", "fuel switching", "emissions-compliant equipment", "non-fossil fuel use", "fuel quality standards"],
    "Lifecycle Carbon Reporting": ["EPD", "ISO 14040", "LCA", "carbon disclosure", "cradle to grave", "cradle to grave analysis", "life cycle assessment", "embodied carbon",  "global warming potential", "whole life carbon", "whole of life emissions"],
    "Low Emission Construction Materials": ["EPD certified", "climate-friendly materials", "green concrete", "green steel", "low GWP products", "low embodied carbon", "low emission materials", "low-carbon concrete", "recycled content", "recycled steel", "sustainable aggregates"],
    "Renewable Energy Systems": ["solar PV", "solar thermal", "on-site renewables", "wind turbine", "clean energy supply"],
    "Renewable Energy Use": ["solar PV", "wind turbine", "renewable sources", "clean energy"],
    "Scope 1 GHG Emissions - Onsite Emissions Reduction Measures": ["low-emission equipment", "electric construction machinery", "no-idling policy", "diesel alternatives"],
    "Scope 2 GHG Emissions - Procurement of Renewable or Clean Electricity": ["renewable electricity", "grid decarbonization", "clean energy supplier", "green power purchase"],
    "Waste Management in Construction": ["construction waste plan", "waste diversion", "recycling targets", "deconstruction waste", "waste audit", "material reuse"],
    "Ecological Impacts": ["biodiversity management plan", "ecological preservation", "flora and fauna protection", "habitat conservation", "ecological corridors", "species impact assessment", "no net loss of biodiversity", "critical habitat avoidance"],
    "Land Use Change": ["controlled site clearance", "habitat protection", "reduced land disturbance", "preservation of existing vegetation", "grading minimization", "sensitive site planning", "ecological buffer zones"],
    "Sustainable Maintenance Planning": ["maintenance plan", "O&M manual", "sustainable operations", "long-term performance", "building tuning"],
    "Air Quality (PM)": ["dust suppression", "PM10 control", "particulate mitigation", "air quality management plan", "water spraying", "dust barriers", "low-dust equipment", "site air monitoring", "fine particle control"],
    "Biological Oxygen Demand (BOD)": ["biological oxygen demand", "BOD limits", "wastewater treatment", "treated discharge", "water effluent quality", "oxygen-demanding substances", "construction wastewater control", "water discharge permit", "EIA water standards"],
    "Chemical Oxygen Demand (COD)": ["chemical oxygen demand", "COD threshold", "treated effluent", "wastewater treatment", "organic load reduction", "water discharge monitoring", "pollutant load control", "construction site effluent standards", "COD testing protocol"],
    "Light Pollution": ["glare control", "shielded lighting", "cut-off luminaires", "dark-sky compliant", "timers or sensors", "reduced spill lighting", "low-impact exterior lighting", "night sky protection"],
    "Noise Pollution": ["noise monitoring", "noise control plan", "sound barriers", "decibel limits", "acoustic insulation", "quiet equipment", "low-noise machinery"],
    "Soil Contamination": ["soil remediation", "contamination prevention", "heavy metals testing", "hazardous waste containment", "soil quality monitoring", "clean soil management", "protective earthworks", "baseline soil assessment", "EIA soil standards"],
    "Suspended Solids": ["suspended solids control", "TSS limits", "sediment traps", "water filtration", "silt fencing", "particle settling tank", "turbidity control", "sedimentation basin", "construction runoff management"],
    "pH Level": ["pH monitoring", "acidity control", "alkalinity limits", "pH adjustment", "neutralization basin", "discharge pH standards", "pH compliant effluent", "pH testing protocol", "pH range compliance"],
    "Stormwater Management": ["stormwater", "runoff", "green infrastructure", "rainwater capture", "stormwater runoff", "permeable pavement", "rain garden", "swale", "detention basin"],
    "Water Harvesting and Efficiency": ["greywater system", "rainwater harvesting", "water recycling", "low-flow fixtures", "potable water reduction"],
    "Indoor Environmental Quality": ["IEQ", "acoustic comfort", "air changes per hour", "comfort metrics", "daylight factor", "daylighting", "indoor air quality", "low VOC", "thermal comfort", "ventilation", "ventilation rate"],
    "Stakeholder Transparency": ["stakeholder communication", "project transparency", "public disclosure", "open reporting", "stakeholder engagement strategy", "information sharing with communities", "project updates to stakeholders", "public access to project data", "transparency commitment clause"],
    "Training and Capacity Building": ["construction workforce training", "capacity building plan", "upskilling program", "technical training for laborers", "site-based skills development", "vocational training", "certified training requirement", "on-the-job training", "education for site workers"],
    "Community Co-Design": ["community engagement", "participatory planning", "stakeholder consultation", "co-design process", "local stakeholder input", "community design workshops", "inclusive planning sessions", "collaborative design", "engagement with affected communities"],
    "Community Engagement": ["co-design", "community feedback", "community input", "feedback sessions", "participatory planning", "public consultation", "public meetings", "stakeholder consultation", "stakeholder input"],
    "Local Employment": ["community employment", "regional workforce", "local hiring", "community-based labor", "regional workforce participation", "employment of local residents", "priority to local workers", "community employment target", "inclusion of local subcontractors", "local job creation"],
    "Gender Inclusion": ["women participation", "female workforce", "gender equity", "women in construction", "female labor participation", "gender-inclusive hiring", "women employment target", "gender-responsive workforce plan", "gender balance in project teams", "inclusion of women-owned subcontractors", "gender diversity reporting"],
    "Gender Responsive Design": ["gender-inclusive design", "safe design for women", "gender-sensitive infrastructure", "female-friendly facilities", "women’s access and safety", "gender-informed site layout", "inclusive public space", "stakeholder feedback on gender needs", "universal design for gender inclusion"],
    "Inclusive Design & Accessibility": ["universal design", "accessible building", "disability access", "barrier-free", "inclusive space"],
    "Worker Health & Safety": ["occupational health and safety", "HSE plan", "personal protective equipment", "PPE compliance", "site safety management", "injury prevention", "safety training", "hazard control", "safety monitoring protocol", "zero accident policy"],
    "Health & Well-being (Indoor Air, Lighting, Acoustic)": ["indoor air quality", "daylighting", "low VOC", "thermal comfort", "acoustic comfort", "ventilation rates"],
    "Cost of Ecosystem Rehabilitation": ["restoration costs", "ecological rehabilitation", "green recovery"],
    "Cost of Relocation": ["resettlement costs", "displacement compensation"],
    "Building Information Modelling (BIM) Use": ["BIM", "BIM brief", "BIM coordination", "BIM execution plan", "building information modelling"],
    "Local Content and Sourcing": ["local procurement", "economic uplift", "regional impact", "local content requirement", "regionally sourced materials", "local suppliers", "community-based sourcing", "preference for local vendors", "domestic procurement target", "locally manufactured inputs", "use of local subcontractors"],
    "Local Economic Benefits": ["local economic development", "support for community enterprises", "local job creation", "inclusive procurement", "regional economic impact", "engagement of local businesses", "SME participation", "community-based suppliers", "local value retention"],
    "Circular Construction Practices": ["design for disassembly", "modular construction", "component reuse", "material passport", "circular design"],
    "Structure Durability": ["design life", "structural longevity", "durable infrastructure", "resilience to degradation", "maintenance-free period", "long-life materials", "infrastructure lifespan", "extended service life", "low-maintenance design"],
    "Lifecycle Cost Analysis": ["lifecycle cost analysis", "LCCA", "whole life costing", "long-term cost evaluation", "cost-benefit analysis", "maintenance cost forecasting", "total cost of ownership", "value for money over lifecycle"]
}

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Sustainability Assessment Tool (with NLP Similarity)")

    uploaded_files = st.file_uploader("Upload PDF documents/contracts", type="pdf", accept_multiple_files=True)

    st.sidebar.header("Manage Keywords")
    st.sidebar.info(
    """
    This application analyzes documents for sustainability indicators based on keyword families from the concept note.
    The keyword list below is derived from the "SUSTAINABLE CONSTRUCTION INDICATOR MATRIX".
    NLP is used to find semantically similar phrases in the document.
    """
    )
    keyword_data = display_keyword_management()

    all_analysis_results = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            raw_text, pages_info, filename = extract_text_and_page_info_from_pdf(uploaded_file)
            if raw_text:
                analysis_results = analyze_document_nlp(raw_text, pages_info, filename, keyword_data)
                all_analysis_results.append(analysis_results)
            else:
                st.error(f"Could not extract text from {uploaded_file.name}. Please ensure it's a readable PDF.")

        if all_analysis_results:
            display_overall_results(all_analysis_results)
            display_detailed_results_nlp(all_analysis_results)


def extract_text_and_page_info_from_pdf(uploaded_file) -> Tuple[str, List[Dict], str]:
    try:
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text_content = ""
        pages_data = [] # Stores end offset of text for each page
        current_total_offset = 0
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            full_text_content += page_text + "\n"
            current_total_offset += len(page_text) + 1 # +1 for the newline
            pages_data.append({"page_number": i + 1, "text_end_offset": current_total_offset})
        return full_text_content, pages_data, uploaded_file.name
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {e}")
        return "", [], uploaded_file.name


def find_page_for_sentence_offset(sentence_start_offset: int, pages_offsets: List[Dict]) -> int:
    """
    Finds the page number for a sentence given its start offset in the full document text
    and a list of page end offsets.
    """
    for page_info in pages_offsets:
        if sentence_start_offset < page_info["text_end_offset"]:
            return page_info["page_number"]
    return pages_offsets[-1]["page_number"] if pages_offsets else 0 # Default to last page or 0


def analyze_document_nlp(document_text: str, pages_info: List[Dict], filename: str, indicator_keywords_matrix: Dict[str, List[str]]) -> Dict:
    detected_indicators_nlp = {}
    
    document_text_lower = document_text.lower() 
    document_sentences_original_case = nltk.sent_tokenize(document_text)
    
    if not document_sentences_original_case:
        return {
            "num_indicators": 0,
            "dimension_coverage": {"Environmental": 0, "Social": 0, "Economic": 0},
            "matched_indicators_nlp": {},
            "ambition_level": "Low",
            "extracted_from": filename
        }

    document_sentences_for_embedding = [s.lower() for s in document_sentences_original_case]

    st.write(f"Embedding sentences for {filename}...")
    doc_sentence_embeddings = np.array(similarity_model.encode(document_sentences_for_embedding, convert_to_tensor=False, show_progress_bar=True))

    st.write(f"Analyzing {filename} for keyword similarity...")
    for indicator, keyword_family in indicator_keywords_matrix.items():
        for keyword_from_matrix in keyword_family:
            keyword_from_matrix_lower = keyword_from_matrix.lower()
            keyword_from_matrix_embedding = np.array(similarity_model.encode(keyword_from_matrix_lower, convert_to_tensor=False))
            
            if doc_sentence_embeddings.ndim == 1:
                 doc_sentence_embeddings = doc_sentence_embeddings.reshape(1, -1)
            if keyword_from_matrix_embedding.ndim == 1:
                 keyword_from_matrix_embedding = keyword_from_matrix_embedding.reshape(1, -1)

            if doc_sentence_embeddings.shape[0] == 0: # No sentences to compare against
                continue

            similarities = cosine_similarity(
                keyword_from_matrix_embedding, 
                doc_sentence_embeddings      
            )[0] 

            for i, score in enumerate(similarities):
                if score >= SIMILARITY_THRESHOLD:
                    matched_sentence_original_case = document_sentences_original_case[i]
                    
                    sentence_start_offset = -1
                    try:
                        sentence_start_offset = document_text.index(matched_sentence_original_case)
                    except ValueError: 
                        try:
                            sentence_start_offset = document_text_lower.find(matched_sentence_original_case.lower())
                        except ValueError:
                             pass # sentence not found, offset remains -1

                    page_num = 0
                    if sentence_start_offset != -1:
                       page_num = find_page_for_sentence_offset(sentence_start_offset, pages_info)
                    
                    if indicator not in detected_indicators_nlp:
                        detected_indicators_nlp[indicator] = []
                    
                    existing_match = False
                    for entry in detected_indicators_nlp[indicator]:
                        if entry['original_keyword'] == keyword_from_matrix and \
                           entry['similar_phrase_in_doc'] == matched_sentence_original_case:
                            existing_match = True
                            if score > entry['similarity_score']:
                                entry['similarity_score'] = float(score)
                                entry['location'] = f"Page {page_num}" if page_num > 0 else "Page unknown"
                            break
                    
                    if not existing_match:
                        detected_indicators_nlp[indicator].append({
                            "original_keyword": keyword_from_matrix, 
                            "similar_phrase_in_doc": matched_sentence_original_case, 
                            "similarity_score": float(score),
                            "location": f"Page {page_num}" if page_num > 0 else "Page unknown",
                            "filename": filename
                        })
            
            if indicator in detected_indicators_nlp:
                detected_indicators_nlp[indicator].sort(key=lambda x: x['similarity_score'], reverse=True)

    num_indicators = len(detected_indicators_nlp)
    dimension_coverage = get_dimension_coverage(detected_indicators_nlp)
    ambition_level = get_ambition_level(num_indicators, dimension_coverage)

    return {
        "num_indicators": num_indicators,
        "dimension_coverage": dimension_coverage,
        "matched_indicators_nlp": detected_indicators_nlp,
        "ambition_level": ambition_level,
        "extracted_from": filename
    }


def get_dimension_coverage(detected_indicators: Dict[str, List]) -> Dict[str, int]:
    environmental_indicators = [
        "Building Energy Efficiency", "Energy Consumption", "Thermal Performance", 
        "Fuel Type for Equipment", "Lifecycle Carbon Reporting", "Low Emission Construction Materials", 
        "Renewable Energy Systems", "Renewable Energy Use", 
        "Scope 1 GHG Emissions - Onsite Emissions Reduction Measures", 
        "Scope 2 GHG Emissions - Procurement of Renewable or Clean Electricity", 
        "Waste Management in Construction", "Ecological Impacts", "Land Use Change", 
        "Sustainable Maintenance Planning", "Air Quality (PM)", "Biological Oxygen Demand (BOD)", 
        "Chemical Oxygen Demand (COD)", "Light Pollution", "Noise Pollution", "Soil Contamination", 
        "Suspended Solids", "pH Level", "Stormwater Management", "Water Harvesting and Efficiency", 
        "Indoor Environmental Quality"
    ]
    social_indicators = [
        "Stakeholder Transparency", "Training and Capacity Building", "Community Co-Design", 
        "Community Engagement", "Local Employment", "Gender Inclusion", "Gender Responsive Design", 
        "Inclusive Design & Accessibility", "Worker Health & Safety", 
        "Health & Well-being (Indoor Air, Lighting, Acoustic)" 
    ]
    economic_indicators = [
        "Cost of Ecosystem Rehabilitation", "Cost of Relocation", 
        "Building Information Modelling (BIM) Use", 
        "Local Content and Sourcing", "Local Economic Benefits", 
        "Circular Construction Practices", 
        "Structure Durability", 
        "Lifecycle Cost Analysis" 
    ]
    
    environmental_count = 0
    social_count = 0
    economic_count = 0

    detected_indicator_names = set(detected_indicators.keys())

    for ind_name in detected_indicator_names:
        if ind_name in environmental_indicators:
            environmental_count +=1
        elif ind_name in social_indicators: 
            social_count +=1
        elif ind_name in economic_indicators: 
            economic_count +=1
            
    return {
        "Environmental": environmental_count,
        "Social": social_count,
        "Economic": economic_count
    }

def get_ambition_level(num_indicators: int, dimension_coverage: Dict[str, int]) -> str:
    num_dimensions_spanned = 0
    if dimension_coverage.get("Environmental", 0) > 0:
        num_dimensions_spanned += 1
    if dimension_coverage.get("Social", 0) > 0:
        num_dimensions_spanned += 1
    if dimension_coverage.get("Economic", 0) > 0:
        num_dimensions_spanned += 1

    if num_indicators >= 10 and num_dimensions_spanned == 3:
        return "High"
    elif 5 <= num_indicators <= 9 and num_dimensions_spanned >= 2:
        return "Medium"
    elif 1 <= num_indicators <= 4:
        return "Low"
    elif num_indicators > 9 and num_dimensions_spanned < 3 :
         return "Medium" 
    else: 
        return "Very Low / Not Assessed"


def display_overall_results(all_analysis_results: List[Dict]):
    st.header("Overall Analysis Summary")

    st.subheader("Overall Ambition Level Distribution")
    if all_analysis_results:
        ambition_levels_present = sorted(list(set(res['ambition_level'] for res in all_analysis_results)))
        
        overall_ambition_data = pd.DataFrame({
            'Level': ambition_levels_present,
            'Count': [sum(1 for res in all_analysis_results if res['ambition_level'] == level) for level in ambition_levels_present]
        })
        if not overall_ambition_data.empty and 'Count' in overall_ambition_data and overall_ambition_data['Count'].sum() > 0 :
            pie_chart = alt.Chart(overall_ambition_data).mark_arc(outerRadius=120).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Level", type="nominal", sort=ambition_levels_present),
                tooltip=['Level', 'Count']
            ).properties(title="Ambition Level Distribution")
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.write("No ambition data to display or all counts are zero.")

        avg_indicators = sum(res['num_indicators'] for res in all_analysis_results) / len(all_analysis_results)
        st.metric(label="Average Number of Indicators Detected", value=f"{avg_indicators:.2f}")

        st.subheader("Overall Dimension Coverage")
        overall_dimension_data = {
            'Environmental': sum(res['dimension_coverage']['Environmental'] for res in all_analysis_results),
            'Social': sum(res['dimension_coverage']['Social'] for res in all_analysis_results),
            'Economic': sum(res['dimension_coverage']['Economic'] for res in all_analysis_results)
        }
        dimension_df = pd.DataFrame(list(overall_dimension_data.items()), columns=['Dimension', 'Count'])
        if not dimension_df.empty:
            chart = alt.Chart(dimension_df).mark_bar().encode(
                x=alt.X('Dimension', sort=None),
                y='Count',
                color=alt.Color('Dimension', legend=None),
                tooltip=['Dimension', 'Count']
            ).properties(title="Combined Dimension Coverage")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No dimension coverage data to display.")
    else:
        st.write("No analysis results to display overall summary.")


def display_detailed_results_nlp(all_analysis_results: List[Dict]):
    st.header("Detailed Results per Document (NLP Similarity)")

    for result in all_analysis_results:
        with st.expander(f"Analysis of: {result['extracted_from']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Ambition Level", value=result['ambition_level'])
            with col2:
                st.metric(label="Number of Unique Indicators Detected", value=result['num_indicators'])

            st.subheader("Dimension Coverage")
            dimension_data = pd.DataFrame(list(result["dimension_coverage"].items()), columns=['Dimension', 'Count'])
            if not dimension_data.empty:
                chart = alt.Chart(dimension_data).mark_bar().encode(
                    x=alt.X('Dimension', sort=None),
                    y='Count',
                    color=alt.Color('Dimension', legend=None),
                    tooltip=['Dimension', 'Count']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No dimension coverage to display for this document.")

            st.subheader("Matched Indicators and Similar Phrases (NLP)")
            if result["matched_indicators_nlp"]:
                data_nlp = []
                for indicator, matches_list in result["matched_indicators_nlp"].items():
                    for match_item in matches_list:
                        data_nlp.append({
                            "Indicator": indicator,
                            "Original Keyword (from Matrix)": match_item["original_keyword"],
                            "Detected Similar Phrase (from Contract)": match_item["similar_phrase_in_doc"],
                            "Similarity (%)": f"{match_item['similarity_score']*100:.2f}%",
                            "Location (Page)": match_item["location"],
                        })
                if data_nlp:
                    df_nlp = pd.DataFrame(data_nlp)
                    st.dataframe(df_nlp)
                else:
                    st.write("No similar phrases found meeting the threshold for this document.")
            else:
                st.write("No indicators matched using NLP for this document.")
        st.markdown("---")


def display_keyword_management():
    st.write("### Indicators and Keyword Families")
    
    if 'editable_keyword_data' not in st.session_state:
        st.session_state.editable_keyword_data = initial_indicator_keywords.copy()

    for indicator, keywords in list(st.session_state.editable_keyword_data.items()):
        with st.expander(indicator):
            keyword_string = ", ".join(keywords)
            new_keyword_str = st.text_area(f"Keywords for {indicator}:", value=keyword_string, key=f"text_area_{indicator}")
            if new_keyword_str != keyword_string:
                st.session_state.editable_keyword_data[indicator] = [k.strip() for k in new_keyword_str.split(",") if k.strip()]
            
            if st.button(f"Remove Indicator: {indicator}", key=f"remove_{indicator}"):
                if indicator in st.session_state.editable_keyword_data:
                    del st.session_state.editable_keyword_data[indicator]
                    st.rerun()

    st.sidebar.subheader("Add New Indicator")
    new_indicator_name = st.sidebar.text_input("New Indicator Name", key="new_indicator_name_input")
    new_indicator_kws = st.sidebar.text_area("Keywords for New Indicator (comma-separated)", key="new_indicator_kws_input")
    if st.sidebar.button("Add Indicator", key="add_new_indicator_button"):
        if new_indicator_name and new_indicator_kws:
            if new_indicator_name not in st.session_state.editable_keyword_data:
                st.session_state.editable_keyword_data[new_indicator_name] = [k.strip() for k in new_indicator_kws.split(",") if k.strip()]
                st.sidebar.success(f"Added indicator: {new_indicator_name}")
                st.rerun()
            else:
                st.sidebar.error("Indicator already exists.")
        else:
            st.sidebar.error("Indicator name and keywords cannot be empty.")
            
    return st.session_state.editable_keyword_data


if __name__ == "__main__":
    main()
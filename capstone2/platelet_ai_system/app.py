"""
AI Enabled Platelet Health Risk Analysis System
Flask Backend Application
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import os
from model_regression import PlateletPredictionModel
from model_classifier import DiseaseClassificationModel
from chatbot import PlateletHealthAssistant

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize AI models
print("🔧 Initializing AI Models...")
platelet_model = PlateletPredictionModel()
disease_model = DiseaseClassificationModel()

# Initialize AI Health Assistant
print("🤖 Initializing Platelet Health AI Assistant...")
health_assistant = PlateletHealthAssistant()

# Train models if they don't exist
if not os.path.exists('platelet_model.joblib'):
    print("Training Platelet Prediction Model...")
    platelet_model.train()

if not os.path.exists('disease_model.joblib'):
    print("Training Disease Classification Model...")
    disease_model.train()

# Load models
platelet_model.model = __import__('joblib').load('platelet_model.joblib') if os.path.exists('platelet_model.joblib') else None
disease_model.model = __import__('joblib').load('disease_model.joblib') if os.path.exists('disease_model.joblib') else None
disease_model.label_encoder = __import__('joblib').load('label_encoder.joblib') if os.path.exists('label_encoder.joblib') else None

print("✓ AI Models Initialized Successfully!")

# Feedback CSV handling
FEEDBACK_FILE = 'feedback.csv'

def init_feedback_file():
    """Initialize feedback CSV file if it doesn't exist."""
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=['Timestamp', 'Age', 'Hemoglobin', 'WBC', 'RBC', 'Predicted_Platelet', 'Condition', 'Helpful', 'Feedback_Text'])
        df.to_csv(FEEDBACK_FILE, index=False)

# Enhanced AI Knowledge Base for interactive chatbot
MEDICAL_KNOWLEDGE_BASE = {
    # Platelet Information
    'platelet': {
        'keywords': ['platelet', 'platelets', 'what are platelets', 'functions of platelets'],
        'response': 'Platelets (thrombocytes) are essential blood cells produced by your bone marrow. Here\'s what you need to know:\n\n📌 **What they do:**\n• Help form blood clots to stop bleeding\n• Repair damaged blood vessels\n• Maintain normal blood vessel walls\n\n🩸 **Normal Range:**\n• 150,000 - 400,000 cells per microliter\n\n⚠️ **Abnormalities:**\n• Below 150,000: Thrombocytopenia (low platelets)\n• Above 400,000: Thrombocytosis (high platelets)\n\n💡 **Why they matter:** Abnormal platelet counts can lead to excessive bleeding or dangerous clotting. Regular monitoring is important for your health.',
        'followup': ['What are symptoms of low platelets?', 'What diseases affect platelet count?', 'How are platelet levels tested?']
    },
    'low_platelet': {
        'keywords': ['low platelet', 'low platelets', 'thrombocytopenia', 'reduced platelet'],
        'response': 'Low platelet count (thrombocytopenia) is a condition where you have fewer than 150,000 platelets per microliter of blood.\n\n🔴 **Symptoms to watch for:**\n• Easy bruising (on skin or under skin)\n• Frequent nosebleeds\n• Bleeding gums\n• Blood in urine or stool\n• Heavy menstrual bleeding\n\n🏥 **Common causes:**\n• Viral infections (dengue, flu, COVID-19)\n• Bone marrow disorders\n• Auto-immune conditions\n• Certain medications\n• Leukemia or other blood cancers\n\n⚡ **What to do:**\n• Consult a hematologist immediately if symptomatic\n• Avoid contact sports and injuries\n• Don\'t take NSAIDs without medical advice\n\n💊 **Treatment:** Depends on cause - may include medications, blood transfusions, or treating underlying condition.',
        'followup': ['What causes dengue-related low platelets?', 'When is low platelet count dangerous?', 'How is thrombocytopenia treated?']
    },
    'high_platelet': {
        'keywords': ['high platelet', 'high platelets', 'thrombocytosis', 'elevated platelet'],
        'response': 'High platelet count (thrombocytosis) means you have more than 400,000 platelets per microliter.\n\n🔴 **Health risks:**\n• Increased risk of blood clots\n• Risk of stroke\n• Risk of heart attack\n• Deep vein thrombosis (DVT)\n\n📊 **Two types:**\n• **Primary:** Caused by bone marrow disorders\n• **Secondary:** Caused by inflammation, infection, or bleeding\n\n🔍 **Common causes:**\n• Iron deficiency\n• Chronic inflammation\n• Bone marrow diseases\n• Recent surgery or bleeding\n• Spleen removal\n\n⚠️ **Important:** High platelet count requires medical evaluation to identify the cause.\n\n💊 **Management:** May include:\n• Treating underlying cause\n• Blood thinning medications\n• Platelet-reducing therapy\n• Regular monitoring',
        'followup': ['What\'s the difference between primary and secondary thrombocytosis?', 'How is high platelet count treated?', 'What complications can occur?']
    },
    'dengue': {
        'keywords': ['dengue', 'dengue fever', 'dengue platelet', 'dengue hemorrhagic'],
        'response': 'Dengue fever is a serious viral infection transmitted by mosquitoes. It\'s particularly known for causing dangerous platelet drops.\n\n🦟 **How it spreads:**\n• Mosquito bites (Aedes mosquitoes)\n• Cannot spread person-to-person (usually)\n\n🤒 **Symptoms (2-7 days after bite):**\n• High fever (104°F+)\n• Severe headache and muscle pain\n• Rash (may appear after fever)\n• Nausea and vomiting\n• Joint pain ("breakbone fever")\n\n🩸 **Platelet Impact:**\n• Dengue often causes thrombocytopenia\n• Platelet count below 50,000 cells/µL is dangerous\n• Can lead to hemorrhagic dengue (severe bleeding)\n\n⚕️ **Treatment:**\n• No specific cure (antiviral)\n• Supportive care (fluids, rest)\n• Pain management (avoid aspirin)\n• Hospitalization if severe\n\n🛡️ **Prevention:**\n• Use mosquito repellent\n• Wear long sleeves/pants in endemic areas\n• Screen windows\n• Dengue vaccines available in some regions',
        'followup': ['How dangerous is dengue fever?', 'What\'s dengue hemorrhagic fever?', 'How long does dengue last?']
    },
    'leukemia': {
        'keywords': ['leukemia', 'blood cancer', 'bone marrow cancer', 'leukaemia'],
        'response': 'Leukemia is a type of cancer affecting blood-forming cells in bone marrow.\n\n🩸 **What happens:**\n• Bone marrow produces abnormal white blood cells\n• Disrupts production of normal blood cells\n• Leads to anemia, infections, low platelets\n\n🔬 **Types:**\n• **Acute Lymphoblastic (ALL):** Rapidly developing \n• **Acute Myeloid (AML):** Rapidly developing\n• **Chronic Lymphocytic (CLL):** Slowly developing\n• **Chronic Myeloid (CML):** Slowly developing\n\n⚠️ **Common symptoms:**\n• Easy bruising and bleeding\n• Unexplained fatigue\n• Frequent infections\n• Night sweats\n• Bone or joint pain\n• Enlarged lymph nodes\n\n🩸 **Blood work findings:**\n• Very low platelet count\n• Abnormal WBC levels\n• Abnormal RBC count\n\n🏥 **Diagnosis:**\n• Blood tests (CBC)\n• Bone marrow biopsy\n• Genetic testing\n\n💊 **Treatment:**\n• Chemotherapy\n• Radiation therapy\n• Stem cell transplant\n• Targeted therapy\n\n📌 **Early detection importance:** Blood tests can detect leukemia early when treatment is most effective.',
        'followup': ['What are the early signs of leukemia?', 'How is leukemia diagnosed?', 'What\'s the survival rate?']
    },
    'infection': {
        'keywords': ['viral infection', 'bacterial infection', 'infection', 'virus', 'bacteria', 'platelet infection'],
        'response': 'Infections significantly affect platelet production. Here\'s what you need to know:\n\n🦠 **How infections affect platelets:**\n• Virus directly affects bone marrow\n• Immune system destroys platelets\n• Inflammation reduces platelet production\n• Severity depends on infection type\n\n🔴 **Viral infections causing low platelets:**\n• **Dengue fever** (most common)\n• **Measles**\n• **COVID-19**\n• **Influenza**\n• **Chickenpox**\n• **HIV**\n• **Hepatitis C**\n\n💔 **Bacterial infections:**\n• Tuberculosis\n• Sepsis\n• Endocarditis\n\n📊 **What happens:**\n• Platelet count usually drops during acute infection\n• Recovers as infection clears\n• Severe infections may cause dangerous drops\n\n⚕️ **What to do:**\n• Get tested to identify infection type\n• Follow medical treatment plan\n• Monitor platelet counts\n• Seek emergency care if severe symptoms\n\n🛡️ **Prevention:**\n• Vaccinations (when available)\n• Good hygiene\n• Avoid infection exposure\n• Healthy immune system',
        'followup': ['How long does platelet recovery take after infection?', 'When is infection-related thrombocytopenia dangerous?', 'Can vaccinations help?']
    },
    'symptoms': {
        'keywords': ['symptoms', 'signs', 'what if', 'i have', 'bruising', 'bleeding', 'nosebleed'],
        'response': 'Platelet-related symptoms vary depending on your platelet count and condition.\n\n🔴 **Low platelet symptoms:**\n• Easy bruising (minimal injury causes large bruises)\n• Petechiae (tiny red/purple dots on skin)\n• Nosebleeds (frequent, hard to stop)\n• Bleeding gums when brushing\n• Blood in urine or stool\n• Heavy menstrual bleeding\n• Prolonged bleeding from cuts\n• Fatigue (if also anemic)\n• Headaches\n\n🔵 **High platelet symptoms:**\n• Blood clot symptoms (leg swelling, pain)\n• Chest pain\n• Shortness of breath\n• Dizziness or fainting\n• Numbness in hands/feet\n• May have no symptoms at all\n\n⚠️ **Emergency symptoms (seek immediate care):**\n• Severe internal bleeding\n• Difficulty breathing\n• Chest pain\n• Vision changes\n• Severe headache with fever\n\n📋 **What to do:**\n1. Document your symptoms and when they started\n2. List any recent infections or illnesses\n3. Schedule blood tests immediately\n4. Avoid behaviors that increase bleeding risk\n5. Consult a hematologist\n\n💡 **Important:** Symptoms don\'t always correlate with platelet count. You can have low platelets without symptoms, or symptoms with normal counts.',
        'followup': ['When should I go to the emergency room?', 'Can stress cause these symptoms?', 'What tests will be done?']
    },
    'treatment': {
        'keywords': ['treatment', 'cure', 'medication', 'medicine', 'therapy', 'how to treat', 'what helps'],
        'response': 'Treatment for platelet abnormalities depends on the cause, severity, and your health status.\n\n💊 **Treatment options:**\n\n**For Low Platelets:**\n• Corticosteroids (reduce immune destruction)\n• IV immunoglobulin (IVIG)\n• Platelet transfusions (emergency use)\n• Spleen removal (in some cases)\n• Medications (romiplostim, eltrombopag)\n• Antibiotics (if infection-related)\n• Treating underlying condition\n\n**For High Platelets:**\n• Antiplatelet drugs (aspirin, clopidogrel)\n• Anticoagulants (warfarin, heparin)\n• Cytoreductive drugs\n• Treating underlying cause\n\n🏥 **Treatment approach:**\n1. **Diagnosis first** - Identify the cause\n2. **Risk assessment** - Determine severity\n3. **Cause-specific therapy** - Treat root cause\n4. **Symptomatic relief** - Manage symptoms\n5. **Monitoring** - Regular follow-ups\n\n⚕️ **Important points:**\n• Treatment is individualized\n• Mild cases may not need treatment\n• Severe cases may need hospitalization\n• Regular blood tests monitor progress\n• Some conditions resolve on their own\n\n🚫 **Things to avoid:**\n• NSAIDs (aspirin, ibuprofen)\n• Blood thinners (unless prescribed)\n• Alcohol (affects platelet function)\n• Contact sports (injury risk)\n\n📌 **Always consult your healthcare provider** - treatment decisions should be made by qualified medical professionals based on your specific condition.',
        'followup': ['Are there natural remedies?', 'How long does treatment take?', 'What are side effects?']
    },
    'test': {
        'keywords': ['test', 'blood test', 'cbc', 'complete blood count', 'platelet test', 'how to check'],
        'response': 'Blood tests are essential for diagnosing and monitoring platelet conditions.\n\n🩸 **Complete Blood Count (CBC):**\nThis is the main test for platelet analysis.\n\n**What it measures:**\n• Platelet count (cells per microliter)\n• Mean Platelet Volume (MPV) - size\n• Platelet Distribution Width (PDW) - variation\n• Red blood cell count\n• White blood cell count\n• Hemoglobin levels\n\n📊 **Normal ranges:**\n• Platelets: 150,000 - 400,000/µL\n• MPV: 7.4 - 10.4 fL\n\n🔬 **Additional tests:**\n• **Blood smear** - visual examination under microscope\n• **Bleeding time test** - how long bleeding takes to stop\n• **Bone marrow biopsy** - if CBC shows abnormalities\n• **Genetic testing** - for inherited disorders\n\n💉 **How blood is collected:**\n1. Arm vein puncture (venipuncture)\n2. Sample collected in special tube\n3. Sent to laboratory\n4. Results in 1-2 days\n\n⏰ **Test preparation:**\n• No special fasting usually required\n• Take at any time of day\n• Some medications may affect results\n• Tell doctor about recent infections\n\n📈 **Regular monitoring:**\n• Baseline measurement\n• Follow-up tests after diagnosis\n• Treatment monitoring\n• Recovery assessment\n\n❓ **Questions to ask doctor:**\n• How often should I be tested?\n• What do my numbers mean?\n• Am I at risk for complications?',
        'followup': ['What do high/low platelet numbers mean?', 'How often should I get tested?', 'Can home tests measure platelets?']
    },
    'prevention': {
        'keywords': ['prevent', 'prevention', 'how to avoid', 'protect', 'healthy platelet'],
        'response': 'While some platelet conditions are due to disease, there are ways to maintain healthy platelets.\n\n🛡️ **Lifestyle measures:**\n\n**Diet:**\n• Eat iron-rich foods (red meat, spinach, legumes)\n• Vitamin B12 sources (eggs, dairy, fish)\n• Folate-rich foods (leafy greens, beans)\n• Vitamin C improves iron absorption\n• Avoid excessive caffeine\n\n**Habits:**\n• Regular exercise (builds blood health)\n• Adequate sleep (supports immunity)\n• Stress management\n• Avoid smoking (affects blood cells)\n• Limit alcohol (affects platelet function)\n\n**Safety:**\n• Use protective gear when needed\n• Avoid contact sports if at risk\n• Be careful with sharp objects\n• Prevent infections (hand hygiene)\n\n**Medical:**\n• Regular blood tests\n• Vaccinations (prevent infections)\n• Avoid medication interactions\n• Report symptoms early\n\n🦟 **Infection prevention:**\n• Mosquito protection (dengue prevention)\n• Good hygiene practices\n• Proper food handling\n• Vaccinations when available\n\n📌 **When to seek help:**\n• Unexplained bruising\n• Frequent nosebleeds\n• Family history of blood disorders\n• Before traveling to high-risk areas\n\n❤️ **General blood health:**\n• Stay hydrated\n• Manage chronic conditions\n• Maintain healthy weight\n• Reduce stress\n• Regular check-ups',
        'followup': ['Should I take supplements?', 'Are blood donations safe for me?', 'What foods boost platelet count?']
    },
    'emergency': {
        'keywords': ['emergency', 'danger', 'serious', 'urgent', 'hospital', 'call ambulance', 'life threatening'],
        'response': '🚨 **SEEK IMMEDIATE EMERGENCY CARE IF YOU EXPERIENCE:**\n\n**Severe bleeding:**\n• Uncontrolled bleeding from any source\n• Large amounts of blood in vomit\n• Black or bloody stools\n• Heavy vaginal bleeding\n• Severe nosebleed\n\n**Neurological symptoms:**\n• Severe headache with fever\n• Confusion or difficulty speaking\n• Vision loss or changes\n• Loss of consciousness\n• Numbness or paralysis\n\n**Cardiovascular symptoms:**\n• Severe chest pain\n• Difficulty breathing\n• Fainting or dizziness\n• Rapid or irregular heartbeat\n\n**Platelet-related emergencies:**\n• Intracranial bleeding signs\n• Severe platelet count drop\n• Signs of sepsis (infection + shock)\n\n☎️ **What to do:**\n1. **Call ambulance or go to ER immediately**\n2. Have recent blood test results ready\n3. List all medications\n4. Describe symptoms and when they started\n5. Tell doctors about platelet disorders\n\n📋 **Information to provide:**\n• Current platelet count\n• Underlying conditions\n• Recent infections\n• Medications and supplements\n• Drug allergies\n\n⚠️ **Don\'t wait** - these symptoms can worsen rapidly\n\n💡 **After emergency:** Follow-up with specialists for proper diagnosis and treatment planning.',
        'followup': ['What are warning signs I shouldn\'t ignore?', 'How to prevent emergencies?', 'What happens in the ER?']
    }
}

conversation_history = {}

def extract_intent(message):
    """Extract the main intent from user message."""
    message_lower = message.lower()
    
    # Check for personal health questions
    personal_health = ['am i alright', 'am i good', 'how am i', 'my health', 'my condition', 'my platelet', 'my blood', 'i have', 'my symptoms']
    if any(phrase in message_lower for phrase in personal_health):
        intent_type = 'personal_health'
    # Check for questions
    elif any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'can', 'should', 'is', 'are']):
        intent_type = 'question'
    elif any(word in message_lower for word in ['help', 'emergency', 'danger', 'severe']):
        intent_type = 'emergency'
    elif any(word in message_lower for word in ['i have', 'my', 'me', 'symptom']):
        intent_type = 'symptom'
    else:
        intent_type = 'general'
    
    return intent_type

def get_personalized_health_assessment(user_data, message):
    """Generate personalized health assessment based on user's blood parameters."""
    if not user_data:
        return {
            'response': 'I\'d love to give you a personalized health assessment! Please first enter your blood parameters (age, hemoglobin, WBC, RBC) in the form above, then I can analyze your specific health condition and give you personalized advice.',
            'followup': ['How do I enter my blood parameters?', 'What tests should I get?', 'When should I see a doctor?']
        }
    
    platelet_count = user_data.get('plateletCount', 0)
    risk_level = user_data.get('riskLevel', 'Unknown')
    top_condition = user_data.get('topCondition', 'Unknown')
    age = user_data.get('age', 0)
    
    message_lower = message.lower()
    
    # Personalized responses based on user's actual data
    if any(phrase in message_lower for phrase in ['am i alright', 'am i okay', 'am i fine', 'am i good']):
        if risk_level == 'Low Risk':
            response = f"Based on your blood analysis, **you're doing well!** 🎉\n\nYour platelet count of {platelet_count:,} cells/µL is within the normal range, and your overall risk level is **{risk_level}**. Your blood parameters indicate good health with no major concerns.\n\n**Keep up the good work!** Continue with regular check-ups and maintain your healthy lifestyle."
        elif risk_level == 'Moderate Risk':
            response = f"Your blood analysis shows **moderate risk** that needs attention. ⚠️\n\nYour platelet count is {platelet_count:,} cells/µL, which falls outside the normal range. The primary concern is **{top_condition}**. While not immediately dangerous, this should be monitored by a healthcare professional.\n\n**Recommendation:** Schedule a consultation with your doctor within the next 1-2 weeks for proper evaluation."
        else:  # High Risk
            response = f"Your blood analysis indicates **high risk** that requires immediate attention! 🚨\n\nYour platelet count of {platelet_count:,} cells/µL is significantly abnormal, suggesting **{top_condition}**. This condition can be serious and may require urgent medical intervention.\n\n**URGENT:** Please consult a healthcare professional immediately - within 24-48 hours if possible."
        
        return {
            'response': response,
            'followup': ['What should I do next?', 'What are the symptoms to watch for?', 'How often should I get tested?']
        }
    
    elif any(phrase in message_lower for phrase in ['my health', 'my condition', 'how is my health']):
        health_status = {
            'Low Risk': 'excellent',
            'Moderate Risk': 'concerning',
            'High Risk': 'serious'
        }.get(risk_level, 'unknown')
        
        response = f"Let me give you a comprehensive overview of your current health condition based on your blood analysis:\n\n🩸 **Your Blood Parameters:**\n• Age: {age} years\n• Platelet Count: {platelet_count:,} cells/µL\n• Risk Level: {risk_level}\n• Primary Condition: {top_condition}\n\n📊 **Health Status:** {health_status.title()}\n\n"
        
        if risk_level == 'Low Risk':
            response += "Your blood work shows normal platelet levels and healthy parameters. No immediate medical intervention is needed, but continue regular monitoring."
        elif risk_level == 'Moderate Risk':
            response += f"Your platelet count suggests {top_condition.lower()}, which requires medical attention. This condition can usually be managed with proper treatment and monitoring."
        else:
            response += f"Your platelet count indicates {top_condition.lower()}, which is a serious condition requiring immediate medical care. Early intervention is crucial for the best outcomes."
        
        return {
            'response': response,
            'followup': ['What treatment options are available?', 'What lifestyle changes should I make?', 'How can I monitor my condition?']
        }
    
    elif any(phrase in message_lower for phrase in ['my platelet', 'my platelets']):
        normal_range = (150000, 400000)
        is_normal = normal_range[0] <= platelet_count <= normal_range[1]
        
        response = f"Let's analyze your platelet count specifically:\n\n📊 **Your Platelet Count:** {platelet_count:,} cells/µL\n\n"
        
        if is_normal:
            response += f"✅ **Normal Range:** Your count is within the healthy range of {normal_range[0]:,} - {normal_range[1]:,} cells/µL.\n\nThis is excellent! Your platelets are functioning normally, which means good clotting ability and overall blood health."
        else:
            if platelet_count < normal_range[0]:
                response += f"⚠️ **Below Normal:** Your count is lower than the healthy range of {normal_range[0]:,} - {normal_range[1]:,} cells/µL.\n\nThis indicates **thrombocytopenia** (low platelets), which can increase bleeding risk. The likely cause is **{top_condition}**."
            else:
                response += f"⚠️ **Above Normal:** Your count is higher than the healthy range of {normal_range[0]:,} - {normal_range[1]:,} cells/µL.\n\nThis indicates **thrombocytosis** (high platelets), which can increase clotting risk. The likely cause is **{top_condition}**."
        
        return {
            'response': response,
            'followup': ['What can cause my platelet level?', 'What symptoms should I watch for?', 'How can I improve my platelet count?']
        }
    
    elif any(phrase in message_lower for phrase in ['what should i do', 'what next', 'what now']):
        if risk_level == 'Low Risk':
            response = "Great news! Your blood analysis shows low risk, so here's what you should do:\n\n✅ **Continue Current Habits:**\n• Maintain your healthy lifestyle\n• Regular exercise and balanced diet\n• Adequate sleep and stress management\n\n📅 **Monitoring:**\n• Annual blood tests to track trends\n• Report any new symptoms immediately\n• Stay hydrated and eat iron-rich foods\n\n🏥 **When to See Doctor:**\n• If you develop unusual symptoms\n• Family history of blood disorders\n• Before major surgery or dental procedures"
        elif risk_level == 'Moderate Risk':
            response = f"Your moderate risk condition requires attention. Here's your action plan:\n\n📞 **Schedule Consultation:**\n• See a hematologist within 1-2 weeks\n• Bring your blood test results\n• Discuss your symptoms and concerns\n\n🔍 **Further Testing:**\n• Additional blood tests may be needed\n• Possibly bone marrow evaluation\n• Genetic testing if indicated\n\n💊 **Management:**\n• Follow prescribed treatment plan\n• Monitor for symptom changes\n• Avoid activities that increase bleeding risk\n\n📊 **Follow-up:**\n• Regular blood tests every 3-6 months\n• Track your platelet counts\n• Report any worsening symptoms immediately"
        else:  # High Risk
            response = f"Your high-risk condition needs urgent medical attention:\n\n🚨 **IMMEDIATE ACTIONS:**\n• Contact healthcare provider today\n• Go to emergency room if severe symptoms\n• Avoid physical activities that could cause injury\n\n📋 **Prepare for Doctor Visit:**\n• List all current symptoms\n• Note when symptoms started\n• Bring all recent blood test results\n• Prepare questions about your condition\n\n🏥 **Emergency Signs:**\n• Severe bleeding or bruising\n• Difficulty breathing\n• Chest pain or palpitations\n• Severe headache or confusion\n\n💡 **While Waiting:** Rest, stay hydrated, avoid aspirin/NSAIDs unless prescribed"
        
        return {
            'response': response,
            'followup': ['What symptoms should I watch for?', 'What questions should I ask my doctor?', 'How often should I get tested?']
        }
    
    # Default personalized response
    return {
        'response': f"I can see you've had a blood analysis done. Based on your results (platelet count: {platelet_count:,} cells/µL, risk level: {risk_level}), I can help answer questions about your specific health condition.\n\nTry asking:\n• 'Am I alright?' - For an overall health assessment\n• 'What should I do next?' - For action recommendations\n• 'Tell me about my platelet count' - For detailed analysis\n• 'How is my health condition?' - For comprehensive overview",
        'followup': ['Am I alright?', 'What should I do next?', 'Tell me about my platelet count']
    }

def get_ai_response(user_message, user_id='general', user_data=None):
    """Generate intelligent AI response using the Platelet Health Assistant."""
    try:
        # Get response from the specialized health assistant
        ai_response = health_assistant.get_response(user_message, user_data)

        # Get follow-up suggestions based on intent
        intent = health_assistant.detect_intent(user_message)
        followup_suggestions = health_assistant.get_followup_suggestions(user_message, intent, user_data)

        return {
            'response': ai_response,
            'followup': followup_suggestions
        }
    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        return {
            'response': 'I apologize, but I encountered an error processing your question. Please try again or rephrase your question.',
            'followup': ['What are normal platelet levels?', 'What causes low platelets?', 'Tell me about dengue fever']
        }

# Routes
@app.route('/')
def home():
    """Home page route."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle patient data and return analysis."""
    try:
        data = request.get_json()
        
        # Extract parameters
        age = float(data.get('age'))
        hemoglobin = float(data.get('hemoglobin'))
        wbc = float(data.get('wbc'))
        rbc = float(data.get('rbc'))
        
        # Validate inputs
        if not (0 < age < 120 and 0 < hemoglobin < 20 and 0 < wbc < 100 and 0 < rbc < 10):
            return jsonify({'error': 'Invalid input parameters'}), 400
        
        # Predict platelet count
        predicted_platelet = platelet_model.predict(age, hemoglobin, wbc, rbc)
        
        if predicted_platelet is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Predict disease/condition
        disease_predictions = disease_model.predict(age, hemoglobin, wbc, rbc, predicted_platelet)
        
        if disease_predictions is None:
            return jsonify({'error': 'Disease prediction failed'}), 500
        
        top_condition = disease_predictions[0]['condition']
        risk_level = disease_model.determine_risk_level(predicted_platelet, top_condition)
        health_explanation = disease_model.get_health_explanation(top_condition, predicted_platelet)
        recommendation = disease_model.get_recommendation(risk_level, top_condition)
        
        # Format platelet count with comma separator
        platelet_formatted = f"{predicted_platelet:,}"
        
        # Build response
        response = {
            'success': True,
            'patientData': {
                'age': age,
                'hemoglobin': hemoglobin,
                'wbc': wbc,
                'rbc': rbc
            },
            'analysis': {
                'plateletCount': predicted_platelet,
                'plateletFormatted': platelet_formatted,
                'riskLevel': risk_level,
                'topCondition': top_condition,
                'conditions': disease_predictions,
                'healthExplanation': health_explanation,
                'recommendation': recommendation
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle AI chat interactions with enhanced intelligence."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'general')
        user_data = data.get('user_data')  # Get user's analysis data
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate AI response with follow-up suggestions and user data
        response_data = get_ai_response(user_message, user_id, user_data)
        ai_response = response_data.get('response', '')
        followup = response_data.get('followup', [])
        
        return jsonify({
            'success': True,
            'userMessage': user_message,
            'aiResponse': ai_response,
            'followupSuggestions': followup,
            'confidence': 'high'
        }), 200
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle feedback submission."""
    try:
        init_feedback_file()
        
        data = request.get_json()
        
        # Extract feedback data
        feedback_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Age': data.get('age', ''),
            'Hemoglobin': data.get('hemoglobin', ''),
            'WBC': data.get('wbc', ''),
            'RBC': data.get('rbc', ''),
            'Predicted_Platelet': data.get('plateletCount', ''),
            'Condition': data.get('condition', ''),
            'Helpful': data.get('helpful', ''),
            'Feedback_Text': data.get('feedbackText', '')
        }
        
        # Append to CSV
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback! Your response helps us improve our AI system.'
        }), 200
    
    except Exception as e:
        print(f"Error in feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AI ENABLED PLATELET HEALTH RISK ANALYSIS SYSTEM")
    print("="*60)
    print("\n🚀 Starting Flask Application...")
    print("📱 Open your browser and navigate to: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)

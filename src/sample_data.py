from datetime import datetime, timedelta
import random

def get_sample_feedback():
    feedback_texts = [
        # Positive Reviews
        "The stay was quite comfortable, and the staff were attentive. The room was clean and well-maintained, and the food was excellent. I felt well-cared for throughout my stay.",
        "I appreciated the prompt response from the nursing staff, especially during the night shifts. The explanations provided by the doctors were clear, and I felt well-informed about my treatment plan.",
        "The nurses were very caring and made sure I was comfortable throughout my stay. The room was a bit small, but it was clean and well-equipped.",
        "The hospital facilities were excellent, and the staff were professional. The care provided was top-notch, and the staff were very supportive.",
        "I felt well-cared for during my stay. The nurses were attentive, and the doctors provided clear explanations. The room was comfortable and clean.",
        "The staff were friendly and professional, and the environment was welcoming. I would recommend this hospital to others.",
        "The medical team was knowledgeable and provided excellent care. The room was spacious and comfortable.",
        "The hospital environment was clean and well-organized. The staff were friendly and attentive.",
        "I appreciated the efforts of the nursing staff to ensure my comfort. The room was clean, and the food was satisfactory.",
        "The care provided was excellent, and the staff were attentive. The facilities were well-maintained.",
        
        # Negative Reviews
        "The room was noisy, and the call button response was slow. The food quality was poor, and I had to wait a long time for assistance.",
        "I was disappointed with the cleanliness of the room. The staff seemed rushed and inattentive, and the noise levels at night were disruptive.",
        "The medical team was not very communicative, and I felt left in the dark about my treatment. The food service was also lacking in variety and taste.",
        "The nurses were often unavailable, and the room was not cleaned regularly. The overall experience was below my expectations.",
        "The hospital environment was chaotic, and the staff were not very helpful. I had to wait a long time for consultations and assistance.",
        "The room was cramped and uncomfortable, and the staff were not very attentive. The food was bland and unappetizing.",
        "I had a negative experience overall. The staff were unprofessional, and the facilities were poorly maintained.",
        "The care provided was inadequate, and the staff were not very supportive. The room was noisy and uncomfortable.",
        "The hospital environment was stressful, and the staff were not very accommodating. I would not recommend this hospital.",
        "The medical team was unresponsive, and the explanations were unclear. The food service was unsatisfactory.",
        
        # Neutral or Ambiguous Reviews
        "The stay was okay, but there were some issues with the Wi-Fi connection. The staff were polite, but the waiting time for assistance was longer than I anticipated.",
        "The room was comfortable, but the noise from the hallway was disruptive at times. The staff were courteous, but the food service could be improved.",
        "I had a mixed experience. The nurses were attentive, but the call button response was inconsistent. The room was clean, but the air conditioning was too cold.",
        "The medical team was knowledgeable, but the explanations were not always clear. The food was satisfactory, but could use more variety.",
        "The hospital facilities were good, but the parking space was limited. The staff were friendly, but the waiting time for consultations was longer than expected.",
        "The stay was average. The staff were neither exceptionally good nor bad. The room was clean, but the amenities were basic.",
        "The experience was neither positive nor negative. The staff were professional, but the facilities were lacking in some areas.",
        "The care provided was adequate, but the staff were not very engaging. The room was comfortable, but the food was mediocre.",
        "The hospital environment was neutral. The staff were polite, but the services were not very efficient.",
        "The medical team was competent, but the communication could be improved. The food service was average.",
    ]

    # Generate 1000 entries with random dates over the past few months
    sample_feedback = []
    for i in range(1000):
        feedback = random.choice(feedback_texts)
        response_time = random.choice(["Always", "Usually", "Sometimes", "Never"])
        bathroom_help = random.choice(["Always", "Usually", "Sometimes", "Never"])
        explanation_clarity = random.choice(["Always", "Usually", "Sometimes", "Never"])
        listening = random.choice(["Always", "Usually", "Sometimes", "Never"])
        courtesy = random.choice(["Always", "Usually", "Sometimes", "Never"])
        recognition = random.choice([
            "Nurse Alice was exceptional.",
            "Dr. Lee for his attentiveness.",
            "No specific recognition.",
            "Nurse Tom for his care.",
            "Nurse Rachel for her empathy.",
            "Nurse Megan for her dedication.",
            "Dr. Martin for clarifying later.",
            "Nurse Lucy for her compassion.",
            "Nurse Jake for his patience.",
            "Nurse Jack for his efforts."
        ])
        timestamp = datetime.now() - timedelta(days=random.randint(0, 180))
        sample_feedback.append((
            f"{1000+i}", feedback, response_time, bathroom_help,
            explanation_clarity, listening, courtesy, recognition,
            timestamp.isoformat()
        ))

    return sample_feedback
from src.multi_agent_pipeline.diffusion_pipeline import run_pipeline

INDIC_CASES = [
    ["hi", "Hindi",     "गाँव में दिवाली मनाता हुआ एक परिवार"],
    ["bn", "Bengali",   "দুর্গাপূজার সময় কলকাতার একটি পাড়ার প্যান্ডেল"],
    ["ta", "Tamil",     "பொங்கல் பண்டிகையில் கோலம் போடும் பெண்கள்"],
    ["te", "Telugu",    "సంక్రాంతి పండుగలో గాలిపటాలు ఎగురవేస్తున్న పిల్లలు"],
    ["mr", "Marathi",   "गणेश चतुर्थीच्या मिरवणुकीत ढोल वाजवणारे लोक"],
    ["gu", "Gujarati",  "નવરાત્રિમાં ગરબા રમતી મહિલાઓ"],
    ["kn", "Kannada",   "ಮೈಸೂರು ದಸರಾ ಮೆರವಣಿಗೆಯಲ್ಲಿ ಅಲಂಕೃತ ಆನೆಗಳು"],
    ["ml", "Malayalam", "ഓണത്തിന് പൂക്കളം ഒരുക്കുന്ന കുടുംബം"],
    ["pa", "Punjabi",   "ਵਿਸਾਖੀ ਦੇ ਮੇਲੇ ਵਿੱਚ ਭੰਗੜਾ ਪਾਉਂਦੇ ਲੋਕ"],
    ["or", "Odia",      "ପୁରୀ ରଥଯାତ୍ରାରେ ରଥ ଟାଣୁଥିବା ଭକ୍ତମାନେ"],
    ["as", "Assamese",  "বিহুৰ সময়ত বিহু নৃত্য কৰা যুৱতীসকল"],
]
 

def run_test():
    """ Run The Above Test cases """
    run = 1
    for test in INDIC_CASES:
        print(f"Run The Test Number: {run}")
        print(f"Language: {test[1]}")
        prompt = test[2]
        print(f"Image Stored in test1/run_{run}.png")
        run = run + 1

if __name__ == "__main__":
    run_test()

    


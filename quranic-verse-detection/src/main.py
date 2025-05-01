# filepath: /quranic-verse-detection/quranic-verse-detection/src/main.py

from ui.interface import UserInterface
from detection.verse_detector import VerseDetector

def main():
    ui = UserInterface()
    ui.display_menu()
    
    detector = VerseDetector()
    detector.load_model()
    
    while True:
        user_input = ui.get_user_input()
        if user_input.lower() == 'exit':
            break
        
        image_path = user_input
        preprocessed_image = detector.preprocess_image(image_path)
        results = detector.detect_verse(preprocessed_image)
        
        ui.show_results(results)

if __name__ == "__main__":
    main()
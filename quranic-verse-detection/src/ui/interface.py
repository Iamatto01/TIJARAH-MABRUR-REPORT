class UserInterface:
    def display_menu(self):
        print("Welcome to Qur'anic Verse Detection!")
        print("Enter the path to an image or type 'exit' to quit.")

    def get_user_input(self):
        return input("Enter image path: ")

    def show_results(self, results):
        print(f"Detection Results: {results}")
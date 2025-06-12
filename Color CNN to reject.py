import random
import datetime
import colorama
from colorama import Fore, init

init()
# Mock model simulating CNN behavior 
class CNNModel:
    def predict(self, stock_symbol):
        # Randomly return a stock action: BUY, SELL, or HOLD
        actions = ['BUY', 'SELL', 'HOLD']
        return random.choice(actions)

# Main stock app logic that uses the CNN model and supports denial
class StockApp:
    def __init__(self, model):
        self.model = model  # Assign the provided model 

    def get_prediction(self, stock_symbol):
        # Get a prediction from the CNN model for the given stock
        prediction = self.model.predict(stock_symbol)
        return prediction

    def execute_action(self, action, stock_symbol):
        # Simulate executing the accepted or overridden action
        print(f"Executing action: {action} for {stock_symbol}")

    def log_denial(self, user_id, stock_symbol, predicted_action, user_action):
        # Log details of the denied CNN decision for auditing/training
        log_entry = {
            "1234": user_id,
            "stock_symbol": stock_symbol,
            "cnn_output": predicted_action,
            "user_decision": user_action,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open("cnn_denials.log", "a") as log_file:
            log_file.write(str(log_entry) + "\n")
        print("Denial logged.")

    def deny_decision(self, user_id, stock_symbol, predicted_action):
        # Called when the user denies the CNN's prediction
        print("CNN decision has been denied.")

        # Ask user for their override action
        user_override = input("Enter your alternative action (BUY/SELL/HOLD): ").strip().upper()

        # Validate and handle the override action
        if user_override in ['BUY', 'SELL', 'HOLD']:
            self.log_denial(user_id, stock_symbol, predicted_action, user_override)
            self.execute_action(user_override, stock_symbol)
        else:
            print("Invalid input. Action aborted.")

    def decision_flow(self, user_id, stock_symbol):
        # Main flow: show prediction and let user accept or deny
        prediction = self.get_prediction(stock_symbol)
        if (prediction == "BUY"):
            print(Fore.GREEN + f"\n recommends: {prediction} for {stock_symbol}")
        if (prediction == "HOLD"):
            print(f"\n recommends: {prediction} for {stock_symbol}")
        if (prediction == "SELL"):
            print(Fore.RED + f"\n recommends: {prediction} for {stock_symbol}")

        
        #print(f"\n recommends: {prediction} for {stock_symbol}")

        # Ask the user to accept or deny the prediction
        decision = input(Fore.WHITE + "Do you want to ACCEPT or DENY this recommendation? (accept/deny): ").strip().lower()

        # Execute based on user input
        if decision == 'accept':
            self.execute_action(prediction, stock_symbol)
        elif decision == 'deny':
            self.deny_decision(user_id, stock_symbol, prediction)
        else:
            print("Invalid input. No action taken.")



if __name__ == "__main__":
    model = CNNModel()  # Instantiate CNN model
    app = StockApp(model)   # Create app instance with the model

    print("Welcome to the Stock App with CNN Decision and Denial")

    # Loop to allow multiple users/stocks until 'exit' is typed
    while True:
        user = input("\nEnter your user ID (or type 'exit' to quit): ").strip()
        if user.lower() == 'exit':
            break

        symbol = input("Enter stock symbol (e.g., AAPL, TSLA): ").strip().upper()
        app.decision_flow(user, symbol)  # Run prediction and decision flow

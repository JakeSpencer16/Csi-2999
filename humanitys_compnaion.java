package real_java_fx;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar;
import javafx.scene.control.ButtonType;
import javafx.scene.control.DialogPane;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
// just a few imports huh?

public class humanitys_compnaion extends Application {

    // User Class to set up some variabls
    class User {
        @SuppressWarnings("FieldMayBeFinal")// I have to do this alot
        private String name;
        private int mood;
        private String problem;

        public User(String name) {
            this.name = name;
            this.mood = 3; // Default mood
            this.problem = "";
        }

        public void setMood(int mood) {
            this.mood = mood;
        }

        public void setProblem(String problem) {
            this.problem = problem;
        }

        public int getMood() {
            return mood;
        }

        public String getProblem() {
            return problem;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }
    }

    // Therapist Class
    class Therapist {
        @SuppressWarnings("FieldMayBeFinal")// I have to do this alot
        private String name;
        @SuppressWarnings("FieldMayBeFinal")// yep
        private ProblemAnalyzer problemAnalyzer;
        @SuppressWarnings("FieldMayBeFinal")// FOUR TIMES
        private MoodBooster moodBooster;
       
        private final List<String> encouragementMessages = List.of(
            // options when there is not a specific awnser
                "You've got this!",
                "Every challenge is an opportunity to grow.",
                "Believe in yourself; you are stronger than you think.",
                "Take things one step at a time.",
                "You are resilient and capable.",
                "Take a deep breath",
                "Grab some cold water",
                "Believe in yourself; I know you can do it!"   
        );
        private final Random random = new Random();

        public Therapist(String name, ProblemAnalyzer problemAnalyzer, MoodBooster moodBooster) {
            this.name = "Sophia";
            this.problemAnalyzer = problemAnalyzer;
            this.moodBooster = moodBooster;
        }

        public String greetUser(User user) {
            return "Hello, " + user.getName() + "! I'm " + this.name + ", and I'm here to help.";
        }

        public List<String> assessProblem(User user) {
            List<String> responses = new ArrayList<>();
            if (user.getProblem().isEmpty()) {
                responses.add("It seems you haven't described your problem yet.");
            } else {
                responses.add("Analyzing your problem: " + user.getProblem());
                List<String> solutions = problemAnalyzer.analyzeProblem(user.getProblem());
                if (solutions.isEmpty()) {
                    responses.add("I'm sorry, I couldn't find any immediate solutions for your problem.");
                    responses.add("Along with a response to hopefully lift your spirits, I will include a alternative.");
                    responses.add("https://www.youtube.com/@Psych2go/videos");
                    responses.add("This channel helps people understand a variety of diffrent mental and physical issues");
                } else {
                    responses.add("Here are some potential solutions, along with a all incompsing piece of advise:");
                    responses.addAll(solutions);
                }
            }
            return responses;
        }

        public String provideEncouragement(User user) {
            int randomIndex = random.nextInt(encouragementMessages.size());
            return encouragementMessages.get(randomIndex);
        }

        public void boostMood(User user) {
            moodBooster.boostMood(user);
        }
    }

    // ProblemAnalyzer Interface
    interface ProblemAnalyzer {
        List<String> analyzeProblem(String problem);
    }

    // MoodBooster Interface
    interface MoodBooster {
        void boostMood(User user);
    }

    // Concrete implementation of ProblemAnalyzer
    class SimpleProblemAnalyzer implements ProblemAnalyzer {
        @Override
        public List<String> analyzeProblem(String problem) {
            List<String> solutions = new ArrayList<>();
            if (problem.toLowerCase().contains("stress")) {
                solutions.add("Findsomething squsihable to grip in your hand.");
                solutions.add("Take short breaks throughout the day.");
            }
            if (problem.toLowerCase().contains("sad")) {
                solutions.add("Talk to someone you trust.");
                solutions.add("Engage in activities you enjoy.");
            }
            if(problem.toLowerCase().contains("long distance relationship")){
                solutions.add("This can be a diffult time for both you and your partner");
                solutions.add("Watch this video series for some ideas as to how to feel more conected while far away:");
                solutions.add("https://youtube.com/playlist?list=PL8ltyl0rAtoPSQSWrcSEZmmNZE-9Nc8kV&si=Z8ORsdNPGz_ounPw");
            }
            if(problem.toLowerCase().contains("focus")){
                solutions.add("Sometimes it can be very hard to keep yourself focused on a task");
                solutions.add("If possible, take a short walk to reorient you mind");
                solutions.add("Sometimes certain types of music can help keep you focused.");
                solutions.add("Music such as this: https://www.youtube.com/@LofiGirl/videos ");
            }
            if(problem.toLowerCase().contains("lonley")){ 
                solutions.add("There can be a lot of ways to counter being lonley, So I shall list a variety of options");
                solutions.add("text a friend or two");
                solutions.add("hug a stuffed animal");
                solutions.add("call a faimly member");
                solutions.add("eating buddy: https://www.youtube.com/watch?v=9mDngl11P3k");
                solutions.add("love one: https://www.youtube.com/@BumbleDeeAudios/videos");
                solutions.add("love two: https://www.youtube.com/@FakeASMR/videos");
            }
            if(problem.toLowerCase().contains("sleep")){
                solutions.add("Slepeing can be hard for a number of reasons, might I sugguest some audio to help you relax?");
                solutions.add("Rain sounds: https://www.youtube.com/watch?v=zbmP1JP9DNc ");
                solutions.add("Loveing: https://www.youtube.com/@covergirlasmr/videos");    
            }
            if(problem.toLowerCase().contains("overwhelmed")){ 
                solutions.add("Slow your breathing, focus on keeping it a nice slow rythym");
                solutions.add("If possible, find a way to tune out outside noise");
                solutions.add("Sing with this song: https://www.youtube.com/watch?v=s2og1KhzHyU");
            }
            if(problem.toLowerCase().contains("sick")){ 
                solutions.add("Know that it is ok to tell your school or work that you are sick");
                solutions.add("Keep track of your body tempreature");
                solutions.add("Drink plenty of hot water and tea, as to help keep your body sustained while sick");
                solutions.add("Avoid all greasy food");
            }
            return solutions;
            
        }
    }

    // Concrete implementation of MoodBooster
    class SimpleMoodBooster implements MoodBooster {
        @Override
        public void boostMood(User user) {
            int currentMood = user.getMood();
            if (currentMood < 5) {
                user.setMood(Math.min(currentMood + 1, 5)); // Increase mood, capped at 5
            }
        }
    }

    private User user;
    private Therapist therapist;
    private TextArea outputArea;
    private TextField problemInputField;
    private Button submitProblemButton;
    private TextField nameInputField;
    

    private final String calmingGreen = "derive(forestgreen, 50%)"; // A slightly lighter shade of forest green

    @Override
    public void start(Stage primaryStage) {
        // --- Name Input Section ---
        Label nameLabel = new Label("Enter your name:");
        nameInputField = new TextField();
        Button nameSubmitButton = new Button("Submit Name");
        VBox nameLayout = new VBox(10, nameLabel, nameInputField, nameSubmitButton);
        nameLayout.setPadding(new Insets(20));
        nameLayout.setStyle("-fx-background-color: " + calmingGreen + ";"); // Set background of the layout
        Scene nameScene = new Scene(nameLayout, 300, 150);

        // --- Main Interaction Section ---
        outputArea = new TextArea();
        outputArea.setEditable(false);
        problemInputField = new TextField();
        problemInputField.setPromptText("Type your problem here...");
        submitProblemButton = new Button("Submit Problem");
        submitProblemButton.setDisable(true);

        VBox mainLayout = new VBox(10, outputArea, problemInputField, submitProblemButton);
        mainLayout.setPadding(new Insets(20));
        mainLayout.setStyle("-fx-background-color: " + calmingGreen + ";"); // Set background of the layout
        Scene mainScene = new Scene(mainLayout, 700, 500);

        primaryStage.setTitle("Humanity's Companion");
        primaryStage.setScene(nameScene);
        primaryStage.show();

        // --- Event Handling ---

        // Name Submission
        nameSubmitButton.setOnAction(e -> {
            String name = nameInputField.getText();
            if (!name.isEmpty()) {
                user = new User(name);
                therapist = new Therapist("Sophia", new SimpleProblemAnalyzer(), new SimpleMoodBooster());
                outputArea.appendText(therapist.greetUser(user) + "\n");
                askForInitialMood(primaryStage, mainScene);
            } else {
                outputArea.appendText("Please enter your name.\n");
            }
        });

        // Problem Submission
        submitProblemButton.setOnAction(e -> {
            String problem = problemInputField.getText();
            if (!problem.isEmpty()) {
                user.setProblem(problem);
                outputArea.appendText("You said: " + problem + "\n");
                for (String response : therapist.assessProblem(user)) {
                    outputArea.appendText(response + "\n");
                }
                outputArea.appendText(therapist.provideEncouragement(user) + "\n");
                therapist.boostMood(user);
                askForMoodUpdate(primaryStage);
                problemInputField.clear();
            }
        });
    }

    private void askForInitialMood(Stage primaryStage, Scene mainScene) {
        Alert moodAlert = new Alert(Alert.AlertType.CONFIRMATION);
        moodAlert.setTitle("Initial Mood");
        moodAlert.setHeaderText("How are you feeling right now (1-5)?\n(1 = Very Bad, 5 = Very Good)");
        DialogPane dialogPane = moodAlert.getDialogPane();
        dialogPane.setStyle("-fx-background-color: " + calmingGreen + ";");

        // Style the header panel
        if (dialogPane.lookup(".header-panel") != null) {
            dialogPane.lookup(".header-panel").setStyle("-fx-background-color: " + calmingGreen + ";");
        }
        // Style the header
        if (dialogPane.getHeader() != null) {
            dialogPane.getHeader().setStyle("-fx-background-color: " + calmingGreen + ";");
        }
        // Style the content area
        dialogPane.setContent(new Label("")); // Add an empty label to style the content area
        dialogPane.lookup(".content").setStyle("-fx-background-color: " + calmingGreen + ";");

        ButtonType buttonTypeOne = new ButtonType("1");
        ButtonType buttonTypeTwo = new ButtonType("2");
        ButtonType buttonTypeThree = new ButtonType("3");
        ButtonType buttonTypeFour = new ButtonType("4");
        ButtonType buttonTypeFive = new ButtonType("5");
        ButtonType buttonTypeCancel = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);

        moodAlert.getButtonTypes().setAll(buttonTypeOne, buttonTypeTwo, buttonTypeThree, buttonTypeFour, buttonTypeFive, buttonTypeCancel);

        Optional<ButtonType> result = moodAlert.showAndWait();
        if (result.isPresent() && result.get() != buttonTypeCancel) {
            try {
                int initialMood = Integer.parseInt(result.get().getText());
                if (initialMood >= 1 && initialMood <= 5) {
                    user.setMood(initialMood);
                    outputArea.appendText("Your initial mood is set to: " + user.getMood() + "\n");
                    submitProblemButton.setDisable(false); // Enable problem submission
                    primaryStage.setScene(mainScene);
                } else {
                    outputArea.appendText("Invalid mood input. Please enter a number between 1 and 5.\n");
                    askForInitialMood(primaryStage, mainScene); // Ask again
                }
            } catch (NumberFormatException e) {
                outputArea.appendText("Invalid input. Please enter a number.\n");
                askForInitialMood(primaryStage, mainScene); // Ask again
            }
        } else {
            outputArea.appendText("You can set your mood later by submitting a problem.\n");
            submitProblemButton.setDisable(false); // Enable problem submission
            primaryStage.setScene(mainScene);
        }
    }

    private void askForMoodUpdate(Stage primaryStage) {
        Alert moodUpdateAlert = new Alert(Alert.AlertType.CONFIRMATION);
        moodUpdateAlert.setTitle("Mood Update");
        moodUpdateAlert.setHeaderText("What is your mood now (1-5)?\n(1 = Very Bad, 5 = Very Good)");
        DialogPane dialogPane = moodUpdateAlert.getDialogPane();
        dialogPane.setStyle("-fx-background-color: " + calmingGreen + ";");
    
        // Style the header panel
        if (dialogPane.lookup(".header-panel") != null) {
            dialogPane.lookup(".header-panel").setStyle("-fx-background-color: " + calmingGreen + ";");
        }
        // Style the header
        if (dialogPane.getHeader() != null) {
            dialogPane.getHeader().setStyle("-fx-background-color: " + calmingGreen + ";");
        }
        // Style the content area
        dialogPane.setContent(new HBox(new Label("New Mood (1-5):"), new TextField())); // Re-add content for styling
        dialogPane.lookup(".content").setStyle("-fx-background-color: " + calmingGreen + ";");
        // Re-set the content with the styled TextField
        TextField moodInput = new TextField();
        moodInput.setStyle("-fx-control-inner-background: derive(white, 80%); -fx-text-fill: black;"); // Keep textbox white-ish
        HBox layout = new HBox(10);
        layout.getChildren().addAll(new Label("New Mood (1-5):"), moodInput);
        moodUpdateAlert.getDialogPane().setContent(layout);
    
        ButtonType okButton = new ButtonType("OK");
        ButtonType cancelButton = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
        moodUpdateAlert.getButtonTypes().setAll(okButton, cancelButton);
    
        // Get the current X-coordinate of the primary stage (main window)
        double primaryStageX = primaryStage.getX();
    
        // Get the Window of the Alert *before* showing it
        Stage dialogStage = (Stage) moodUpdateAlert.getDialogPane().getScene().getWindow();
    
        // Set the X position of the dialog
        dialogStage.setX(primaryStageX - 360); // Adjust the offset as needed
    
        // Show the alert and wait for a response
        Optional<ButtonType> result = moodUpdateAlert.showAndWait();
    
        if (result.isPresent() && result.get() == okButton) {
            String newMoodText = moodInput.getText();
            try {
                int newMood = Integer.parseInt(newMoodText);
                if (newMood >= 1 && newMood <= 5) {
                    user.setMood(newMood);
                    outputArea.appendText("Your mood is now: " + user.getMood() + "\n");
                } else {
                    outputArea.appendText("Invalid mood input. Please enter a number between 1 and 5.\n");
                    askForMoodUpdate(primaryStage); // Ask again
                }
            } catch (NumberFormatException e) {
                outputArea.appendText("Invalid input. Please enter a number.\n");
                askForMoodUpdate(primaryStage); // Ask again
            }
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}

// in future, add more options, more resources, and possibly an vtuberlike model to help with visual comfort.
//There's also ways to optimize code such as instead of putting solutions as a list, you can then just write something like, "sad.txt" and then read the file that way
//so, in summary of last line, just spam .txt/.csv files instead of a bunch of responses.add calls.
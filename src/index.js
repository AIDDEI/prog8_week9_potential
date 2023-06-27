// Globals
const nn = ml5.neuralNetwork({ task: 'regression', debug: true });
let trainData;
let testData;

let correct = 0;
let incorrect = 0;
let totalAmount = 0;

let saveButton = document.getElementById("saveButton");

// Data
const csvFile = "data/fifa23.csv";

// Load the CSV file
function loadData() {
    Papa.parse(csvFile, {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: results => sortData(results.data),
    });
}

// Sort the data and create the train data and test data
function sortData(data) {
    // Shuffle
    data.sort(() => (Math.random() - 0.5));

    // Create the trainData and testData
    trainData = data.slice(0, Math.floor(data.length * 0.8));
    testData = data.slice(Math.floor(data.length * 0.8) + 1);

    // Set the total amount of the test data
    totalAmount = 2000;

    // Add all players of the trainData to the neural network
    for (let player of trainData) {
        nn.addData({ overall: player.overall, value: player.value, age: player.age }, { potential: player.potential });
    }

    // Normalize the data
    nn.normalizeData();

    // Start training
    startTraining();
}

// Start the training
function startTraining() {
    nn.train({ epochs: 25 }, () => finishedTraining());
}

// Training is finished
function finishedTraining() {
    console.log("Finished Training");
    makePrediction();
}

// Test the model
async function makePrediction() {
    for (let i = 0; i < totalAmount; i++) {
        const testPlayer = { overall: testData[i].overall, value: testData[i].value, age: testData[i].age };
        const prediction = await nn.predict(testPlayer);
        let predictedPotential = Math.round(prediction[0].potential);

        if (!predictedPotential || !testData[i].potential) {
            continue;
        } else {
            if (predictedPotential === testData[i].potential || predictedPotential == testData[i].potential + 1 || predictedPotential == testData[i].potential + 2 || predictedPotential == testData[i].potential - 1 || predictedPotential == testData[i].potential - 2) {
                console.log(`De voorspelling is juist, want het voorspelde potentieel ${predictedPotential} is gelijk aan het echte potentieel ${testData[i].potential}!`);
                correct++;
            } else {
                console.log(`De voorspelling is onjuist, want het voorspelde potentieel is ${predictedPotential}, terwijl het echte potentieel ${testData[i].potential} is.`);
                incorrect++;
            }
        }
    }

    await calculateAccuracy();
}

// Calculate the accuracy
async function calculateAccuracy() {
    let accuracy = (correct / totalAmount) * 100;
    console.log(`De nauwkeurigheid is ${accuracy}%: er zijn in het totaal ${totalAmount} values, waarvan er ${correct} correct zijn en ${incorrect} incorrect.`);
    saveButton.addEventListener("click", saveModel);
}

// Save the model
function saveModel() {
    nn.save();
}

// Execute
loadData();
document.addEventListener("DOMContentLoaded", function() {
  const submitButton = document.getElementById('btn-submit');
//  const fetchButton = document.getElementById('btn-fetch');
  disableSubmitButton('btn-submit');
  document.querySelectorAll('.controlButton').forEach(button => {
    button.disabled = false;
  });
  initializeQuiz();
});

function disableSubmitButton(buttonId) {
    const button = document.getElementById(buttonId);
    button.disabled = true;
}

function enableSubmitButton(buttonId) {
    const button = document.getElementById(buttonId);
    button.disabled = false;
    button.style.display = 'block';
    button.style.backgroundColor = '#5d2f94';
    button.value = 'Submit';
    button.addEventListener('click', function(event) {
        updateSpreadsheet(user_data);
    });
}

document.getElementById('quiz-main').addEventListener('click', function(event) {
  event.preventDefault();

  document.querySelectorAll('.controlButton').forEach(button => {
    button.disabled = false;
  });
  disableSubmitButton('btn-submit');
  resetQuiz();
  initializeQuiz();
});

let currentIndex = 0;
let questions = [];
let user_data = [];

function initializeQuiz() {
  fetch('static/data/OCEAN_test.txt')
    .then(response => response.text())
    .then(data => {
      questions = data.split('\n').filter(question => question.trim() !== '');
      displayQuestion(currentIndex);
    })
    .catch(error => console.error('Error fetching the text file:', error));

  document.querySelectorAll('.controlButton').forEach(button => {
    button.addEventListener('click', handleButtonClick);
  });
}


function displayQuestion(currentIndex) {
    question = questions[currentIndex].split('\t');
    len = questions.length;
    q_type = question[0];
    explanation = question[1];
    question= question[2];
    if (currentIndex < len) {
        document.getElementById('q_id').value = q_type;
        document.getElementById('questionText').value = question;
        document.getElementById('explanation').value = explanation;
        user_data.push(q_type);
    }
    if (currentIndex == questions.length - 1) {
        document.getElementById('questionText').value = 'No more questions.Click on submit button';
        document.querySelectorAll('.controlButton').forEach(button => {
            button.disabled = true;
        });
        enableSubmitButton('btn-submit');
    }
}

function handleButtonClick(event) {
    let dataIndex = event.currentTarget.dataset.index;
    user_data.push(dataIndex);
    if (currentIndex < questions.length) {
        currentIndex++;
        displayQuestion(currentIndex);
    }
}
function resetQuiz() {
    currentIndex = 0;
    user_data = [];
    questions = [];

    document.querySelectorAll('.controlButton').forEach(button => {
        let newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
    });
}

function updateSpreadsheet(user_data) {
    const jsonData = JSON.stringify(user_data);

    fetch('/saveData', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        console.log('Data saved successfully');
    })
    .catch(error => console.error('Error:', error));
    console.log('End of line in quizjs');
}




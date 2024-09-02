//function fetchScores() {
//    fetch('/getScores')
//        .then(response => response.json())
//        .then(data => {
//            // Update your scores on the page with the data
////            document.getElementById('suggestedCareers').innerText = JSON.stringify(data.suggestedCareers);
//            document.getElementById('numerical-aptitude').innerText = data.scores[Numerical_Aptitude];
//            document.getElementById('spatial-aptitude').innerText = JSON.stringify(scores.Spatial_Aptitude);
//            document.getElementById('perceptual-aptitude').innerText = JSON.stringify(scores.Perceptual_Aptitude);
//            document.getElementById('abstract-reasoning').innerText = JSON.stringify(scores.Abstract_Reasoning);
//            document.getElementById('verbal-reasoning').innerText = JSON.stringify(scores.Verbal_Reasoning);
//
//        });
//}

// Call fetchScores periodically, e.g., every 10 seconds
//setInterval(fetchScores, 10000);
//-------------------------------------------------
//const socket = io();
//
//socket.on('update_data', function(data) {
//
//    console.log(data);
//    document.querySelector('.numerical-aptitude').textContent = data.scores.Numerical_Aptitude;
//    document.querySelector('.numerical-aptitude-bar').textContent = data.scores.Numerical_Aptitude;
//    document.querySelector('.val').textContent = data.scores.Spatial_Aptitude;
//    document.querySelector('.spatial-aptitude-bar').textContent = data.scores.Spatial_Aptitude;
//    document.querySelector('.perceptual-aptitude').textContent = data.scores.Perceptual_Aptitude;
//    document.querySelector('.perceptual-aptitude-bar').textContent = data.scores.Perceptual_Aptitude;
//    document.querySelector('.abstract-reasoning').textContent = data.scores.Abstract_Reasoning;
//    document.querySelector('.abstract-reasoning-bar').textContent = data.scores.Abstract_Reasoning;
//    document.querySelector('.verbal-reasoning').textContent = data.scores.Verbal_Reasoning;
//    document.querySelector('.verbal-reasoning-bar').textContent = data.scores.Verbal_Reasoning;
//});
//---------------------------------------------------------------
//$(document).ready(function() {
//    $.getJSON('scores.json', function(data) {
//        console.log(data);
//
//        $('.numerical-aptitude').html(data.scores.Numerical_Aptitude);
//        $('.numerical-aptitude-bar').html(data.scores[Numerical_Aptitude]);
//        $('.val').html(data.scores.Spatial_Aptitude);
//        $('.spatial-aptitude-bar').html(data.scores.Spatial_Aptitude);
//        $('.perceptual-aptitude').html(data.scores.Perceptual_Aptitude);
//        $('.perceptual-aptitude-bar').html(data.scores.Perceptual_Aptitude);
//        $('.abstract-reasoning').html(data.scores.Abstract_Reasoning);
//        $('.abstract-reasoning-bar').html(data.scores.Abstract_Reasoning);
//        $('.verbal-reasoning').html(data.scores.Verbal_Reasoning);
//        $('.verbal-reasoning-bar').html(data.scores.Verbal_Reasoning);
//    }).fail(function() {
//        console.log('An error occurred while fetching scores.json');
//    }
//
//$.getJSON('suggested_career.json', function(data) {
//    console.log(data);
//    const careersList = $('#careers-list');
//    careersList.empty();
//    data.suggested_careers.forEach(function(career) {
//        const listItem = $('<li>').text(career);
//        careersList.append(listItem);
//    }).fail(function() {
//        console.log('An error occurred while fetching scores.json');
//    }
//}
//}
//---------------------------------------------------------
//document.addEventListener("DOMContentLoaded", function() {
//
//document.getElementById('quiz-main').addEventListener('click', function(event) {
//setTimeout(myFunction, 5000);
//updateUI();
//console.log("End of line in updateUI")
//})
//})
//-------------------------------------------------------------
//function updateUI() {
//    fetch('/updateData', {
//        method: 'GET'
//    })
//    .then(response => response.json())
//    .then(data => {
//        // Update scores
//        document.getElementById('numerical-aptitude').textContent = data.scores.Numerical_Aptitude;
//        document.getElementById('numerical-aptitude-bar').textContent = data.scores.Numerical_Aptitude;
//        document.getElementById('spatial-aptitude').textContent = data.scores.Spatial_Aptitude;
//        document.getElementById('spatial-aptitude-bar').textContent = data.scores.Spatial_Aptitude;
//        document.getElementById('perceptual-aptitude').textContent = data.scores.Perceptual_Aptitude;
//        document.getElementById('perceptual-aptitude-bar').textContent = data.scores.Perceptual_Aptitude;
//        document.getElementById('abstract-reasoning').textContent = data.scores.Abstract_Reasoning;
//        document.getElementById('abstract-reasoning-bar').textContent = data.scores.Abstract_Reasoning;
//        document.getElementById('verbal-reasoning').textContent = data.scores.Verbal_Reasoning;
//        document.getElementById('verbal-reasoning-bar').textContent = data.scores.Verbal_Reasoning;
//
//        alert("Finished")
//
//        // Update careers
//        const careersList = document.getElementById('careers-list');
//        careersList.innerHTML = ''; // Clear the existing list
//        data.suggested_careers.forEach(function(career) {
//            const listItem = document.createElement('li');
//            listItem.textContent = career;
//            careersList.appendChild(listItem);
//        });
//    })
//    .catch(error => console.error('Error:', error));
//}
//
//// Fetch the suggested careers from the JSON file
//fetch('suggested_careers.json')
//    .then(response => {
//        if (!response.ok) {
//            throw new Error('Network response was not ok to bring suggested careers');
//        }
//        return response.json();
//    })
//    .then(data => {
//        // Update the careers list with the fetched data
//        updateCareersList(data.suggested_careers);
//    })
//    .catch(error => console.error('Error fetching the careers data:', error));
//
//
//// Fetch the suggested careers from the JSON file
//fetch('scores.json')
//    .then(response => {
//        if (!response.ok) {
//            throw new Error('Network response was not ok for fetching scores.json');
//        }
//        return response.json();
//    })
//    .then(data => {
//        // Update the careers list with the fetched data
//        updateCareersList(data.suggested_careers);
//    })
//    .catch(error => console.error('Error fetching the score data:', error));

//====================================================

//
//        document.addEventListener("DOMContentLoaded", function() {
//            var socket = io();
//
//            socket.on('update_scores', function(scores) {
//                document.getElementById('numerical-aptitude').innerText = scores['Numerical_Aptitude'] + '%';
//                document.getElementById('numerical-aptitude-bar').style.width = scores['Numerical_Aptitude'] ;
//
//                document.getElementById('spatial-aptitude').innerText = scores['Spatial_Aptitude'] + '%';
//                document.getElementById('spatial-aptitude-bar').style.width = scores['Spatial_Aptitude'] ;
//            });
//        });
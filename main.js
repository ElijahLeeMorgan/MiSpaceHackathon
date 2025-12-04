// Forecast buttons functionality
const buttons = document.querySelectorAll('button');
const forecasts = document.querySelectorAll('.forecast-img');

buttons.forEach((btn, index) => {
  btn.addEventListener("click", () => {

    // Hide all forecast items
    forecasts.forEach(f => f.style.display = "none");
    forecasts[index].style.display = "block";

    // Remove active class from all buttons
    buttons.forEach(b => b.classList.remove("active"));

    // Add active class to the clicked button
    btn.classList.add("active");
  });
});

/* Runs before first paint (loaded in <head>) so there is no flash. Light is the default. */
(function () {
  var t = "light";
  try { if (localStorage.getItem("oscc-theme") === "dark") t = "dark"; } catch (e) {}
  document.documentElement.setAttribute("data-theme", t);
})();

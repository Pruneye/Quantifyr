document.addEventListener("DOMContentLoaded", function() {
  var main = document.querySelector('.md-main');
  var sidebar = document.querySelector('.md-sidebar--primary .md-sidebar__scrollwrap');
  if (main && sidebar) {
    main.addEventListener('scroll', function() {
      if (main.scrollTop === 0) {
        sidebar.scrollTop = 0;
      }
    });
  }
});

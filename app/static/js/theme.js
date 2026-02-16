/**
 * Theme toggle functionality
 * Handles dark/light mode switching with localStorage persistence
 */

(function () {
    'use strict';

    const toggleButton = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;

    /**
     * Get the current theme
     * @returns {string} 'dark' or 'light'
     */
    function getCurrentTheme() {
        return htmlElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }

    /**
     * Set the theme
     * @param {string} theme - 'dark' or 'light'
     */
    function setTheme(theme) {
        if (theme === 'dark') {
            htmlElement.setAttribute('data-theme', 'dark');
        } else {
            htmlElement.removeAttribute('data-theme');
        }
        localStorage.setItem('theme', theme);
    }

    /**
     * Toggle between dark and light themes
     */
    function toggleTheme() {
        const currentTheme = getCurrentTheme();
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
    }

    /**
     * Initialize theme on page load
     * Checks localStorage first, falls back to OS preference
     */
    function initTheme() {
        const savedTheme = localStorage.getItem('theme');

        if (savedTheme) {
            // Use saved preference
            setTheme(savedTheme);
        } else {
            // Check OS preference
            const prefersDark = window.matchMedia(
                '(prefers-color-scheme: dark)'
            ).matches;
            setTheme(prefersDark ? 'dark' : 'light');
        }
    }

    /**
     * Listen for OS theme changes
     * Only applies if user hasn't set a manual preference
     */
    function listenForOSThemeChanges() {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

        mediaQuery.addEventListener('change', function (e) {
            // Only auto-switch if user hasn't manually set a preference
            if (!localStorage.getItem('theme')) {
                setTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    // Initialize
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleTheme);
    }

    initTheme();
    listenForOSThemeChanges();
})();

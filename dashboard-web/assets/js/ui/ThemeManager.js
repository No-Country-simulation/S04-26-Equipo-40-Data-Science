/**
 * ThemeManager.js — ConversaAI Dashboard
 *
 * SOLID — Single Responsibility:
 * Manages ONLY dark/light theme switching. Nothing else.
 *
 * Persists preference in localStorage.
 * Emits 'theme:changed' event via EventBus for Chart.js re-rendering.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    const STORAGE_KEY = 'conversaai-theme';

    const ThemeManager = {
        /** Initialize theme from stored preference */
        init() {
            const stored = localStorage.getItem(STORAGE_KEY);
            const theme  = stored || (
                window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
            );
            this._apply(theme);
            this._bindEvents();
        },

        /** Return current theme */
        current() {
            return document.documentElement.getAttribute('data-theme') || 'dark';
        },

        /** Toggle between dark and light */
        toggle() {
            const next = this.current() === 'dark' ? 'light' : 'dark';
            this._apply(next);
        },

        /** Apply a specific theme */
        _apply(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem(STORAGE_KEY, theme);
            this._updateIcon(theme);
            ConversaAI.EventBus.emit('theme:changed', { theme });
        },

        /** Update sun/moon icon */
        _updateIcon(theme) {
            const icon = document.getElementById('themeIcon');
            if (!icon) return;
            icon.className = theme === 'dark'
                ? 'fa-solid fa-sun'
                : 'fa-solid fa-moon';
        },

        /** Bind all theme toggle buttons */
        _bindEvents() {
            const handlers = [
                document.getElementById('themeToggleBtn'),
                document.getElementById('themeToggleNav')
            ];
            handlers.forEach(el => {
                if (el) el.addEventListener('click', e => {
                    e.preventDefault();
                    this.toggle();
                });
            });
        }
    };

    ConversaAI.ThemeManager = ThemeManager;

})(window);

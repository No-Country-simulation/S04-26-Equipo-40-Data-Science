/**
 * TabManager.js — ConversaAI Dashboard
 *
 * SOLID — Single Responsibility:
 * Manages ONLY tab activation state, URL-free navigation, and breadcrumb.
 *
 * Uses EventBus (Observer) to notify other modules when a tab changes,
 * so charts can lazy-render on first view.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /** Tab metadata */
    const TABS = {
        overview:        { label: 'Resumen General',       panel: 'tab-overview' },
        models:          { label: 'Comparativa de Modelos', panel: 'tab-models' },
        frustration:     { label: 'Análisis de Frustración', panel: 'tab-frustration' },
        recommendations: { label: 'Recomendaciones',        panel: 'tab-recommendations' }
    };

    const TabManager = {
        _current: null,

        /** Initialize tabs and bind nav links */
        init() {
            // Bind sidebar nav links
            Object.keys(TABS).forEach(tabId => {
                const link = document.getElementById(`nav-${tabId}`);
                if (!link) return;
                link.addEventListener('click', e => {
                    e.preventDefault();
                    this.activateTab(tabId);
                    // Close sidebar on mobile
                    ConversaAI.EventBus.emit('sidebar:close');
                });
            });

            // Default active tab
            this.activateTab('overview');
        },

        /** Activate a tab by id */
        activateTab(tabId) {
            if (this._current === tabId) return;
            const meta = TABS[tabId];
            if (!meta) { console.warn(`[TabManager] Unknown tab: ${tabId}`); return; }

            const prev = this._current;
            this._current = tabId;

            // Update panels
            document.querySelectorAll('.tab-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            const panel = document.getElementById(meta.panel);
            if (panel) panel.classList.add('active');

            // Update nav links
            document.querySelectorAll('.sidebar__link[data-tab]').forEach(link => {
                link.classList.toggle('active', link.dataset.tab === tabId);
                link.setAttribute('aria-current', link.dataset.tab === tabId ? 'page' : 'false');
            });

            // Update breadcrumb
            const bc = document.getElementById('breadcrumbPage');
            if (bc) bc.textContent = meta.label;

            // Notify other modules
            ConversaAI.EventBus.emit('tab:changed', { tabId, prev, label: meta.label });
        },

        getCurrent: () => null // will be overridden by closure
    };

    // Patch getCurrent to use closure variable
    TabManager.getCurrent = () => TabManager._current;

    ConversaAI.TabManager = TabManager;

})(window);

/**
 * app.js — ConversaAI Dashboard
 *
 * PATTERN: Facade
 * Single entry point that wires together all subsystems:
 * DataService → ChartFactory → UIController → TabManager → ThemeManager
 *
 * SOLID — Dependency Inversion:
 * DashboardApp depends on abstractions (DataService, ChartFactory, etc.),
 * never on concrete implementations directly.
 *
 * SOLID — Single Responsibility:
 * app.js only orchestrates initialization and event wiring.
 * It does NOT contain business logic, data, or rendering details.
 */
(function (global) {
    'use strict';

    const {
        EventBus, DataService, ChartFactory, UIController,
        TabManager, ThemeManager, FilterContext, FilterStrategies
    } = global.ConversaAI;

    /* ----------------------------------------------------------------
       Chart instance registry (prevents canvas conflicts on re-render)
       ---------------------------------------------------------------- */
    const _charts = {};

    /* ----------------------------------------------------------------
       FilterContext — Strategy Pattern wired up
       ---------------------------------------------------------------- */
    const langFilterCtx = FilterContext(FilterStrategies.Language);

    /* ----------------------------------------------------------------
       Get singleton data service
       ---------------------------------------------------------------- */
    const ds = DataService.getInstance();

    /* ----------------------------------------------------------------
       Chart render helpers
       (Facade delegates to ChartFactory + concrete renderers)
       ---------------------------------------------------------------- */

    function renderOverviewCharts(lang) {
        const dist     = ds.getSentimentDist(lang);
        const intents  = ds.getIntentDist(lang);
        const timeline = ds.getTimeline(lang);

        _charts.sentiment = ChartFactory.create(
            'doughnut', 'sentimentChart',
            ChartFactory.buildSentimentConfig(dist),
            _charts.sentiment
        );

        _charts.intent = ChartFactory.create(
            'bar', 'intentChart',
            ChartFactory.buildIntentConfig(intents),
            _charts.intent
        );

        _charts.timeline = ChartFactory.create(
            'line', 'timelineChart',
            ChartFactory.buildTimelineConfig(timeline),
            _charts.timeline
        );

        UIController.updateSentimentBadge(lang);
    }

    function renderModelsCharts(lang) {
        // Apply language filter via Strategy pattern
        const allModels = ds.getModels();
        const models = langFilterCtx.execute(allModels, { lang });

        _charts.accuracy = ChartFactory.create(
            'bar', 'accuracyChart',
            ChartFactory.buildAccuracyConfig(models),
            _charts.accuracy
        );

        _charts.latency = ChartFactory.create(
            'bar', 'latencyChart',
            ChartFactory.buildLatencyConfig(models),
            _charts.latency
        );

        _charts.radar = ChartFactory.create(
            'radar', 'radarChart',
            ChartFactory.buildRadarConfig(models),
            _charts.radar
        );

        _charts.f1 = ChartFactory.create(
            'bar', 'f1Chart',
            ChartFactory.buildF1Config(models),
            _charts.f1
        );

        UIController.renderModelsTable(models);
    }

    function renderFrustrationCharts(lang) {
        const churn      = ds.getChurnDist();
        const frustIntent = ds.getFrustrationByIntent();
        const frustTime  = ds.getFrustrationTimeline(lang);

        _charts.churn = ChartFactory.create(
            'doughnut', 'churnChart',
            ChartFactory.buildChurnConfig(churn),
            _charts.churn
        );

        _charts.frustIntent = ChartFactory.create(
            'bar', 'frustIntentChart',
            ChartFactory.buildFrustIntentConfig(frustIntent),
            _charts.frustIntent
        );

        _charts.frustTimeline = ChartFactory.create(
            'line', 'frustTimelineChart',
            ChartFactory.buildFrustTimelineConfig(frustTime),
            _charts.frustTimeline
        );
    }

    function renderRecommendationsTab() {
        UIController.renderRecommendations(ds.getRecommendations());
        UIController.renderConfigCards(ds.getModels());
    }

    /* ----------------------------------------------------------------
       Lazy chart rendering — only render when tab is first visited
       ---------------------------------------------------------------- */
    const _rendered = { overview: false, models: false, frustration: false, recommendations: false };

    function renderTab(tabId, lang) {
        if (tabId === 'overview') {
            renderOverviewCharts(lang);
            _rendered.overview = true;
        }
        if (tabId === 'models' && !_rendered.models) {
            renderModelsCharts(lang);
            _rendered.models = true;
        }
        if (tabId === 'frustration' && !_rendered.frustration) {
            renderFrustrationCharts(lang);
            _rendered.frustration = true;
        }
        if (tabId === 'recommendations' && !_rendered.recommendations) {
            renderRecommendationsTab();
            _rendered.recommendations = true;
        }
    }

    function refreshCurrentTab(lang) {
        const current = TabManager.getCurrent();
        if (current) renderTab(current, lang);
    }

    /* ----------------------------------------------------------------
       Language filter dropdown
       ---------------------------------------------------------------- */
    function initLangFilter() {
        const select = document.getElementById('langFilter');
        if (!select) return;

        select.addEventListener('change', () => {
            const lang = select.value;
            ds.setLang(lang);
            // Re-render models tab even if already visited
            if (TabManager.getCurrent() === 'models') {
                renderModelsCharts(lang);
            }
        });
    }

    /* ----------------------------------------------------------------
       EventBus subscriptions (Observer pattern in action)
       ---------------------------------------------------------------- */

    // When data filter changes → refresh overview charts
    EventBus.on('data:filtered', ({ lang }) => {
        if (TabManager.getCurrent() === 'overview') {
            renderOverviewCharts(lang);
        }
        UIController.showToast(`Filtro aplicado: ${lang === 'all' ? 'Todos los idiomas' : lang.toUpperCase()}`);
    });

    // When tab changes → lazy-render that tab
    EventBus.on('tab:changed', ({ tabId }) => {
        renderTab(tabId, ds.getLang());
    });

    // When theme changes → no chart re-render needed (CSS vars handle colors)
    // (Chart tooltips are styled via Chart.js defaults set at init)

    /* ----------------------------------------------------------------
       Apply Chart.js global defaults (dark theme defaults)
       ---------------------------------------------------------------- */
    function applyChartDefaults() {
        if (typeof Chart === 'undefined') return;
        Chart.defaults.color              = '#94a3b8';
        Chart.defaults.font.family        = "'Inter', sans-serif";
        Chart.defaults.font.size          = 12;
        Chart.defaults.borderColor        = 'rgba(255,255,255,0.05)';
        Chart.defaults.plugins.tooltip.cornerRadius = 10;
    }

    /* ----------------------------------------------------------------
       DashboardApp — Facade init()
       ---------------------------------------------------------------- */
    const DashboardApp = {
        init() {
            applyChartDefaults();

            // Initialize subsystems in dependency order
            ThemeManager.init();
            UIController.init(ds);
            TabManager.init();   // triggers 'tab:changed' → renderOverviewCharts
            initLangFilter();

            console.info('[ConversaAI] Dashboard initialized ✅');
            console.info('[ConversaAI] Registered models:', ds.getModels().map(m => m.name));
        }
    };

    /* ----------------------------------------------------------------
       Boot on DOMContentLoaded
       ---------------------------------------------------------------- */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => DashboardApp.init());
    } else {
        DashboardApp.init();
    }

    // Expose for debugging
    global.ConversaAI.App = DashboardApp;

})(window);

/**
 * UIController.js — ConversaAI Dashboard
 *
 * SOLID — Single Responsibility:
 * Manages DOM rendering (tables, rec cards, config cards, toasts, sidebar).
 * Does NOT handle routing, data fetching, or chart creation.
 *
 * SOLID — Dependency Inversion:
 * Depends on DataService and ModelRegistry abstractions, not raw data.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /** Safe get element helper */
    function $(id) { return document.getElementById(id); }

    /* ----------------------------------------------------------------
       Status badge HTML helper
       ---------------------------------------------------------------- */
    function _statusBadge(status) {
        const MAP = {
            active:   { cls: 'active',   label: 'En Producción' },
            testing:  { cls: 'testing',  label: 'En Pruebas' },
            baseline: { cls: 'baseline', label: 'Baseline' }
        };
        const s = MAP[status] || MAP.baseline;
        return `<span class="status-badge status-badge--${s.cls}">${s.label}</span>`;
    }

    /* ----------------------------------------------------------------
       Language badge helper
       ---------------------------------------------------------------- */
    function _langBadge(lang) {
        const MAP = {
            'es':    { flag: '🇪🇸', label: 'Español',   cls: 'orange' },
            'pt':    { flag: '🇧🇷', label: 'Portugués',  cls: 'green' },
            'es+pt': { flag: '🌐',  label: 'ES + PT',    cls: 'blue' }
        };
        const l = MAP[lang] || MAP['es+pt'];
        return `<span class="badge badge--${l.cls}">${l.flag} ${l.label}</span>`;
    }

    /* ----------------------------------------------------------------
       Accuracy mini-bar helper
       ---------------------------------------------------------------- */
    function _accBar(value, color = '#6366f1') {
        return `
        <div class="acc-bar">
            <div class="acc-bar__track">
                <div class="acc-bar__fill" style="width:${value}%;background:${color}"></div>
            </div>
            <span class="acc-bar__label" style="color:${color}">${value}%</span>
        </div>`;
    }

    /* ----------------------------------------------------------------
       Models Table
       ---------------------------------------------------------------- */
    function _renderModelsTable(models) {
        const tbody = $('modelsTableBody');
        if (!tbody) return;

        const colorMap = {
            'tfidf-logreg':   { acc: '#64748b', f1: '#64748b' },
            'robertuito-es':  { acc: '#f97316', f1: '#f97316' },
            'bertimbau-pt':   { acc: '#10b981', f1: '#10b981' },
            'xlmr-multilang': { acc: '#6366f1', f1: '#6366f1' }
        };

        tbody.innerHTML = models.map(m => {
            const c = colorMap[m.id] || { acc: '#6366f1', f1: '#6366f1' };
            return `
            <tr>
                <td>
                    <strong style="color:var(--text-primary)">${m.name}</strong>
                    <br><small style="color:var(--text-muted)">${m.useCase}</small>
                </td>
                <td><code>${m.architecture}</code></td>
                <td>${_langBadge(m.lang)}</td>
                <td>${_accBar(m.accuracy, c.acc)}</td>
                <td><span style="color:var(--text-secondary);font-weight:600">${m.recall}%</span></td>
                <td>${_accBar(m.f1Score, c.f1)}</td>
                <td>
                    <span style="color:var(--text-primary);font-weight:700">${m.latencyMs}ms</span>
                    <br><small style="color:var(--text-muted)">${m.trainingSamples.toLocaleString()} samples</small>
                </td>
                <td>${_statusBadge(m.status)}</td>
            </tr>`;
        }).join('');
    }

    /* ----------------------------------------------------------------
       Recommendations List
       ---------------------------------------------------------------- */
    function _renderRecommendations(recs) {
        const list = $('recList');
        if (!list) return;

        const iconMap = {
            critical: 'fa-triangle-exclamation',
            high:     'fa-fire',
            medium:   'fa-wrench',
            low:      'fa-circle-info'
        };

        list.innerHTML = recs.map((rec, i) => `
        <div class="rec-item rec-item--${rec.priority}" role="listitem"
             style="animation-delay:${i * 60}ms">
            <div class="rec-item__icon">
                <i class="fa-solid ${rec.icon || iconMap[rec.priority]}"></i>
            </div>
            <div class="rec-item__body">
                <p class="rec-item__title">${rec.title}</p>
                <p class="rec-item__action">${rec.action}</p>
            </div>
            <span class="rec-priority-badge rec-priority-badge--${rec.priority}">
                ${rec.priority.toUpperCase()}
            </span>
        </div>`).join('');
    }

    /* ----------------------------------------------------------------
       Config Cards (model technical details)
       ---------------------------------------------------------------- */
    function _renderConfigCards(models) {
        const grid = $('configGrid');
        if (!grid) return;

        grid.innerHTML = models.map(m => `
        <div class="config-card">
            <p class="config-card__name">${m.name}</p>
            <code class="config-card__repo">${m.repo}</code>
            <div class="config-card__meta">
                <div class="config-card__meta-item">
                    <span class="config-card__meta-key">Arquitectura</span>
                    <span class="config-card__meta-val">${m.architecture.split(' ')[0]}</span>
                </div>
                <div class="config-card__meta-item">
                    <span class="config-card__meta-key">Idioma</span>
                    <span class="config-card__meta-val">${m.lang.toUpperCase()}</span>
                </div>
                <div class="config-card__meta-item">
                    <span class="config-card__meta-key">Precisión</span>
                    <span class="config-card__meta-val" style="color:var(--emerald)">${m.accuracy}%</span>
                </div>
                <div class="config-card__meta-item">
                    <span class="config-card__meta-key">Latencia</span>
                    <span class="config-card__meta-val">${m.latencyMs}ms</span>
                </div>
                <div class="config-card__meta-item">
                    <span class="config-card__meta-key">Estado</span>
                    <span class="config-card__meta-val">${m.status}</span>
                </div>
            </div>
        </div>`).join('');
    }

    /* ----------------------------------------------------------------
       Toast Notification
       ---------------------------------------------------------------- */
    let _toastTimer = null;

    function _showToast(message, duration = 3000) {
        const toast   = $('toast');
        const msgEl   = $('toastMessage');
        if (!toast || !msgEl) return;

        msgEl.textContent = message;
        toast.hidden = false;
        toast.classList.remove('toast--out');

        clearTimeout(_toastTimer);
        _toastTimer = setTimeout(() => {
            toast.classList.add('toast--out');
            setTimeout(() => { toast.hidden = true; }, 300);
        }, duration);
    }

    /* ----------------------------------------------------------------
       Sidebar mobile toggle
       ---------------------------------------------------------------- */
    function _initSidebar() {
        const sidebar  = document.getElementById('sidebar');
        const overlay  = document.getElementById('sidebarOverlay');
        const hamburger = document.getElementById('hamburgerBtn');
        const closeBtn = document.getElementById('sidebarCloseBtn');

        function open()  {
            sidebar.classList.add('open');
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
        function close() {
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }

        if (hamburger) hamburger.addEventListener('click', open);
        if (closeBtn)  closeBtn.addEventListener('click', close);
        if (overlay)   overlay.addEventListener('click', close);

        // EventBus: close on tab change (mobile)
        ConversaAI.EventBus.on('sidebar:close', close);
    }

    /* ----------------------------------------------------------------
       Sentiment badge in overview tab
       ---------------------------------------------------------------- */
    function _updateSentimentBadge(lang) {
        const badge = $('sentimentLangBadge');
        if (!badge) return;
        const MAP = { all: 'Todos', es: '🇪🇸 ES', pt: '🇧🇷 PT' };
        badge.textContent = MAP[lang] || 'Todos';
    }

    /* ----------------------------------------------------------------
       Export recommendations as CSV
       ---------------------------------------------------------------- */
    function _exportAllDataJson(ds) {
        const snapshot = {
            exportedAt:  new Date().toISOString(),
            langFilter:  ds.getLang(),
            sentiment:   ds.getSentimentDist(),
            intents:     ds.getIntentDist(),
            timeline:    ds.getTimeline(),
            frustration: {
                timeline: ds.getFrustrationTimeline(),
                byIntent: ds.getFrustrationByIntent()
            },
            churn:       ds.getChurnDist(),
            recommendations: ds.getRecommendations(),
            models: {
                all:    ds.getModels(),
                active: ds.getFilteredModels()
            }
        };

        const json  = JSON.stringify(snapshot, null, 2);
        const blob  = new Blob([json], { type: 'application/json;charset=utf-8;' });
        const url   = URL.createObjectURL(blob);
        const a     = document.createElement('a');
        a.href = url;
        a.download = `conversaai_reporte_${new Date().toISOString().slice(0,10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    function _exportRecsCsv(recs) {
        const header = 'Prioridad,Título,Acción';
        const rows = recs.map(r =>
            `"${r.priority.toUpperCase()}","${r.title}","${r.action}"`
        );
        const csv = [header, ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url;
        a.download = `conversaai_recomendaciones_${new Date().toISOString().slice(0,10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /* ----------------------------------------------------------------
       Public UIController API
       ---------------------------------------------------------------- */
    const UIController = {
        init(dataService) {
            this._ds = dataService;
            _initSidebar();

            // Bind export buttons
            const exportBtn     = $('exportBtn');
            const exportRecsBtn = $('exportRecsBtn');

            if (exportBtn) exportBtn.addEventListener('click', () => {
                _exportAllDataJson(dataService);
                _showToast('📊 Reporte JSON descargado con todos los datos del dashboard.');
            });
            if (exportRecsBtn) exportRecsBtn.addEventListener('click', () => {
                _exportRecsCsv(dataService.getRecommendations());
                _showToast('📄 CSV de recomendaciones descargado.');
            });
        },

        renderModelsTable(models) { _renderModelsTable(models); },
        renderRecommendations(recs) { _renderRecommendations(recs); },
        renderConfigCards(models) { _renderConfigCards(models); },
        updateSentimentBadge(lang) { _updateSentimentBadge(lang); },
        showToast(msg, duration)   { _showToast(msg, duration); }
    };

    ConversaAI.UIController = UIController;

})(window);

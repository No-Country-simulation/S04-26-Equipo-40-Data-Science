/**
 * LineChart.js — ConversaAI Dashboard
 * SOLID — Liskov Substitution + Single Responsibility
 * Concrete renderer for line/area charts. Same render() interface as BarChart.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};
    ConversaAI.Charts = ConversaAI.Charts || {};

    function _baseOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutCubic' },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                        boxWidth: 12,
                        boxHeight: 12,
                        borderRadius: 3,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(13, 20, 38, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    boxPadding: 4
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10, family: 'Inter' },
                        maxTicksLimit: 10,
                        maxRotation: 0
                    }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#94a3b8', font: { size: 11, family: 'Inter' } },
                    beginAtZero: true
                }
            }
        };
    }

    const LineChart = {
        /**
         * Render (or update) a line/area chart.
         * @param {string}  canvasId - Canvas element id
         * @param {Object}  config   - { labels, datasets, options? }
         * @param {Object}  existing - Existing Chart.js instance (optional)
         * @returns {Chart}
         */
        render(canvasId, config, existing) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) { console.warn(`[LineChart] Canvas #${canvasId} not found`); return null; }
            if (existing) { existing.destroy(); }

            const opts = Object.assign({}, _baseOptions(), config.options || {});

            return new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: { labels: config.labels, datasets: config.datasets },
                options: opts
            });
        }
    };

    ConversaAI.Charts.LineChart = Object.freeze(LineChart);

})(window);

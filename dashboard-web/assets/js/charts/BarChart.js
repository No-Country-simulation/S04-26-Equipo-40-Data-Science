/**
 * BarChart.js — ConversaAI Dashboard
 *
 * PATTERN: Concrete implementation of IChartRenderer
 * SOLID — Liskov Substitution: Implements the same render(config) interface
 * as all other chart implementations, so ChartFactory can substitute any.
 * SOLID — Single Responsibility: Only renders bar charts.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};
    ConversaAI.Charts = ConversaAI.Charts || {};

    /**
     * Shared Chart.js base options for bar charts (dark & light aware).
     * @param {boolean} horizontal
     * @returns {Object}
     */
    function _baseOptions(horizontal = false) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: horizontal ? 'y' : 'x',
            animation: { duration: 700, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13, 20, 38, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    displayColors: true,
                    boxPadding: 4
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#94a3b8', font: { size: 11, family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#94a3b8', font: { size: 11, family: 'Inter' } }
                }
            }
        };
    }

    const BarChart = {
        /**
         * Render (or update) a bar chart.
         * @param {string}  canvasId  - Canvas element id
         * @param {Object}  config    - { labels, datasets, options?, horizontal? }
         * @param {Object}  existing  - Existing Chart.js instance to update (optional)
         * @returns {Chart}
         */
        render(canvasId, config, existing) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) { console.warn(`[BarChart] Canvas #${canvasId} not found`); return null; }

            // Destroy and recreate if updating (simplest approach, avoids Chart.js update bugs)
            if (existing) { existing.destroy(); }

            const opts = Object.assign({}, _baseOptions(config.horizontal || false), config.options || {});

            return new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: config.labels,
                    datasets: config.datasets
                },
                options: opts
            });
        }
    };

    ConversaAI.Charts.BarChart = Object.freeze(BarChart);

})(window);

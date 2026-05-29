/**
 * FilterStrategy.js — ConversaAI Dashboard
 *
 * PATTERN: Strategy
 * Defines a family of interchangeable filtering algorithms.
 * The DashboardApp selects the appropriate strategy at runtime
 * without modifying the filtering logic.
 *
 * SOLID — Open/Closed:
 * New filter strategies can be added without changing existing ones.
 *
 * SOLID — Single Responsibility:
 * Each strategy class handles ONE filtering concern.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /* ----------------------------------------------------------------
       IFilterStrategy — "interface" (enforced by duck-typing)
       Each strategy must implement: apply(data, params) => filtered data
       ---------------------------------------------------------------- */

    /**
     * LanguageFilterStrategy
     * Filters data items that have a 'lang' field matching the target.
     */
    const LanguageFilterStrategy = {
        name: 'language',
        /**
         * @param {Object[]} data   - Array of items with .lang property
         * @param {Object}   params - { lang: 'all'|'es'|'pt' }
         * @returns {Object[]}
         */
        apply(data, params) {
            const { lang = 'all' } = params;
            if (lang === 'all') return data;
            return data.filter(item =>
                item.lang === lang || item.lang === 'es+pt'
            );
        }
    };

    /**
     * StatusFilterStrategy
     * Filters items by their operational status.
     */
    const StatusFilterStrategy = {
        name: 'status',
        /**
         * @param {Object[]} data   - Array of items with .status property
         * @param {Object}   params - { statuses: string[] }
         * @returns {Object[]}
         */
        apply(data, params) {
            const { statuses = [] } = params;
            if (!statuses.length) return data;
            return data.filter(item => statuses.includes(item.status));
        }
    };

    /**
     * AccuracyThresholdFilterStrategy
     * Filters models that meet a minimum accuracy threshold.
     */
    const AccuracyThresholdFilterStrategy = {
        name: 'accuracy-threshold',
        /**
         * @param {Object[]} data   - Array of model items with .accuracy
         * @param {Object}   params - { minAccuracy: number }
         * @returns {Object[]}
         */
        apply(data, params) {
            const { minAccuracy = 0 } = params;
            return data.filter(item => (item.accuracy || 0) >= minAccuracy);
        }
    };

    /**
     * FilterContext
     * Holds the selected strategy and executes it.
     * Consumers program against FilterContext, not concrete strategies.
     */
    function FilterContext(strategy) {
        let _strategy = strategy;

        return {
            setStrategy(newStrategy) {
                _strategy = newStrategy;
            },
            execute(data, params) {
                if (!_strategy) throw new Error('[FilterContext] No strategy set.');
                return _strategy.apply(data, params);
            }
        };
    }

    ConversaAI.FilterStrategies = Object.freeze({
        Language:          LanguageFilterStrategy,
        Status:            StatusFilterStrategy,
        AccuracyThreshold: AccuracyThresholdFilterStrategy
    });

    ConversaAI.FilterContext = FilterContext;

})(window);

(window.webpackJsonpf8a8ad94_4cf3_4a19_a76b_1cec9da00219_0_1_0=window.webpackJsonpf8a8ad94_4cf3_4a19_a76b_1cec9da00219_0_1_0||[]).push([[2],{SCym:function(e,t,n){"use strict";n.r(t);var a=n("UWqr"),i=n("cKBR"),r=n("2q6Q"),o=n("bKG0"),s=n("LDNF"),c=function(){function e(e){var t=this;this._hasBeenActivated=!1,this._isActive=!1,this._onActivatedCallbacks=[],this._onSearchBoxActivatedCallback=function(e){t._isActive=!0,r._EngagementLogger.log({name:"SuiteNavSearchBox.FocusIn",isIntentional:!0,extraData:{scenario:Object(i.e)(e)}}),t._hasBeenActivated||(r._PerformanceLogger.mark("SuiteNavSearchBox_FocusIn"),t._hasBeenActivated=!0),t._onActivatedCallbacks.forEach(function(e){return e()})},this._onSearchBoxDeactivatedCallback=function(e){t._isActive=!1,r._EngagementLogger.log({name:"SuiteNavSearchBox.FocusOut",isIntentional:!0,extraData:{scenario:Object(i.e)(e)}})},this._serviceScope=e}return e.prototype.configureSearchBox=function(e){var t=this;if(this._isSearchBoxEnabled()){if(this._initializedPromise)return this._initializedPromise;var n=new r._QosMonitor("SuiteNavSearchBox.Configure");return this._initializedPromise=Object(s.e)(this._serviceScope).then(function(a){a.OnSearchBoxActivated(t._onSearchBoxActivatedCallback.bind(t,e.preloadedData.clientSideApplicationId)),a.OnSearchBoxDeactivated(t._onSearchBoxDeactivatedCallback.bind(t,e.preloadedData.clientSideApplicationId)),n.writeSuccess()}).catch(function(e){n.writeUnexpectedFailure(void 0,e)}),this._initializedPromise}return Promise.resolve()},e.prototype.onActivated=function(e){var t=this;return-1===this._onActivatedCallbacks.indexOf(e)&&(this._onActivatedCallbacks=this._onActivatedCallbacks.concat(e)),this._isActive&&e(),function(){t._onActivatedCallbacks=t._onActivatedCallbacks.filter(function(t){return t!==e})}},e.prototype.updateSearchBoxConfiguration=function(e){e.searchBoxConfigurationCallback&&e.searchBoxConfigurationCallback()},e.prototype._isSearchBoxEnabled=function(){return o.e.isSearchBoxInHeaderFlighted()},e.serviceKey=a.ServiceKey.create("sp-suite-nav:SearchBoxProvider",e),e}();t.default=c}}]);
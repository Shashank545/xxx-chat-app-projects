(window.webpackJsonp_8496636c_2300_4915_abef_20de64c98d8b_1_19_0=window.webpackJsonp_8496636c_2300_4915_abef_20de64c98d8b_1_19_0||[]).push([[1],{"+3E/":function(e,t,n){"use strict";n.d(t,"c",function(){return i}),n.d(t,"t",function(){return r}),n.d(t,"e",function(){return o}),n.d(t,"r",function(){return s}),n.d(t,"a",function(){return c}),n.d(t,"o",function(){return d}),n.d(t,"s",function(){return l}),n.d(t,"d",function(){return u}),n.d(t,"l",function(){return f}),n.d(t,"n",function(){return p}),n.d(t,"i",function(){return m});var a=n("UWqr");function i(){return a._SPKillSwitch.isActivated("dd3dba4e-4baa-4bd3-95bf-64ffae9b7b51")}function r(){return a._SPKillSwitch.isActivated("a4a3d2f4-ad0e-4684-adee-a5ceec193558")}function o(){return a._SPKillSwitch.isActivated("e212149a-997e-4aa9-aa89-8be796153d3e")}function s(){return a._SPKillSwitch.isActivated("ca6e64a5-3cde-4350-8a45-a21de378cc60")}function c(){return a._SPKillSwitch.isActivated("e0bd3586-89a3-466b-8ee0-db1f9457fe99")}function d(){return a._SPKillSwitch.isActivated("1a423106-78a1-4238-ba44-dd5f7f4138df")}function l(){return a._SPKillSwitch.isActivated("fed6018f-7453-47fc-aecb-2d9505b62bb4")}function u(){return a._SPKillSwitch.isActivated("733da2bd-f34d-4979-8c7b-df7342291267")}function f(){return a._SPKillSwitch.isActivated("043feae9-5a56-40ff-b175-6f7f813507ff")}function p(){return a._SPKillSwitch.isActivated("5a350491-8d32-46a3-a4c4-fdce19a5f73e")}function m(){return a._SPKillSwitch.isActivated("e6246a29-f6af-41de-88b7-3f023102181b")}},NUTh:function(e,t,n){"use strict";n.r(t),n.d(t,"DeferredAadTokenProvider",function(){return f});var a=n("17wl"),i=n("UWqr"),r=n("ut3N"),o=n("bB1B"),s=n("MGiw"),c=n("wRfT"),d=n("+3E/"),l=n("4Dzj"),u=function(e){function t(t,n){var a=e.call(this,t)||this;return a.code=n,a}return Object(a.__extends)(t,e),t}(Error),f=function(){function e(e,t,n,a,r,o,s){this._oboFirstPartyTokenCallback=o,this._oboThirdPartyTokenCallback=s,i.Validate.isNonemptyString(a.aadInstanceUrl,"aadInstanceUrl"),i.Validate.isNonemptyString(a.aadTenantId,"aadTenantId"),i.Validate.isNonemptyString(a.redirectUri,"redirectUri"),i.Validate.isNonemptyString(a.servicePrincipalId,"servicePrincipalId"),this._defaultConfiguration=a,this._oboConfiguration=r,this._tokenAcquisitionEvent=e,this.onBeforeRedirectEvent=t,this.popupEvent=n,this._failedTokenRequests=new Map}return e.prototype.getToken=function(e,t){void 0===t&&(t=!0);var n=t&&"object"==typeof t?t:{useCachedToken:t,authenticationScheme:l.e.BEARER};return this._defaultConfiguration.servicePrincipalId===i.Guid.empty.toString()?Promise.reject(new Error(o.t)):this._getTokenInternal(e,this._defaultConfiguration,n)},e.prototype._getTokenData=function(e,t,n){void 0===t&&(t=!0),void 0===n&&(n=!1);var a=t&&"object"==typeof t?t:{useCachedToken:t,skipLoggingAndDisableRedirects:n,claims:void 0,authenticationScheme:l.e.BEARER};if(this._defaultConfiguration.servicePrincipalId===i.Guid.empty.toString())return Promise.reject(new Error(o.t));if(this._shouldUseMsalBrowserTokenProvider(this._defaultConfiguration))return this._getMsalBrowserTokenProvider(this._defaultConfiguration).then(function(t){return t.getTokenData(e,a)});if(this._shouldUseMsalTokenProvider(this._defaultConfiguration))return this._getMsalTokenProvider(this._defaultConfiguration).then(function(t){return t.getTokenData(e,a)});throw new Error("Getting token response not supported for the current token provider")},e.prototype._getTokenInternal=function(e,t,n){var a=this;void 0===n&&(n=!0);var i,o=n&&"object"==typeof n?n:{useCachedToken:n,authenticationScheme:l.e.BEARER},c=new r._QosMonitor("AadTokenProvider.GetAppTokenTimePerf");if(!this._shouldTokenBeRequested(e))throw c.writeExpectedFailure("Token already requested and failed"),Object(d.t)()?Error("Token request previously failed"):new u("Token request previously failed","TokenRequestPreviouslyFailed");var f=t||this._defaultConfiguration;if(this._shouldUseOboTokenExchange()&&this._oboConfiguration)i=this._getOboTokenProvider(f,this._oboConfiguration,this._oboFirstPartyTokenCallback,this._oboThirdPartyTokenCallback).then(function(t){return a._getToken(t,e,o)});else{if(this._shouldUseMsalBrowserTokenProvider(f))return this._getMsalBrowserTokenProvider(f).then(function(t){return a._getToken(t,e,o)});i=this._shouldUseMsalTokenProvider(f)?this._getMsalTokenProvider(f).then(function(t){return a._getToken(t,e,o)}):this._getImplicitTokenProvider(f).then(function(t){return a._getToken(t,e,o)})}return i.then(function(t){var n=f.servicePrincipalId===s.e.PRE_AUTHORIZED_APP_PRINCIPAL_ID,a={isInternal:n};return n&&(a.name=e),c.writeSuccess(a),t}).catch(function(t){throw Object(d.e)()?c.writeUnexpectedFailure():c.writeUnexpectedFailure(void 0,t),a._addFailedRequest(e),t})},e.prototype._addFailedRequest=function(e){Object(d.c)()||this._failedTokenRequests.set(e,new Date(Date.now()))},e.prototype._shouldTokenBeRequested=function(e){if(!Object(d.c)()){var t=this._failedTokenRequests.get(e);if(t)return Date.now()-t.getTime()>3e5}return!0},e.prototype._getToken=function(e,t,n){return e.getToken(t,n)},Object.defineProperty(e.prototype,"tokenAcquisitionEvent",{get:function(){return this._tokenAcquisitionEvent},enumerable:!1,configurable:!0}),e.prototype._shouldUseMsalTokenProvider=function(e){var t=!!i.Guid.tryParse(e.aadSessionId),n=t||""===e.aadSessionId;if(!t){var a=new r._QosMonitor("DeferredAadTokenProvider._shouldUseMsalTokenProvider"),o={aadSessionId:e.aadSessionId};n&&(e.aadSessionId=""),a.writeUnexpectedFailure("isAadSessionIdValidGuid",void 0,o)}return n},e.prototype._shouldUseMsalBrowserTokenProvider=function(e){return!(!Object(c.n)()&&!e.enableClaimChallenges)&&this._isFirstParty(e.servicePrincipalId)},e.prototype._getAdalAuthContextManager=function(e){return this._authContextManager||(this._authContextManager=Promise.all([n.e(11),n.e(2)]).then(n.bind(null,"z1gO")).then(function(e){return new e.AdalAuthContextManager})),this._authContextManager},e.prototype._getMsalTokenProvider=function(e){var t=this;return Object(c.o)()?Promise.all([n.e(0),n.e(6)]).then(n.bind(null,"EZRm")).then(function(n){return t._isFirstParty(e.servicePrincipalId)?(t._firstPartyMsalTokenProvider||(t._firstPartyMsalTokenProvider=new n.MsalTokenProvider(e)),t._firstPartyMsalTokenProvider):(t._thirdPartyMsalTokenProvider||(t._thirdPartyMsalTokenProvider=new n.MsalTokenProvider(e)),t._thirdPartyMsalTokenProvider)}):Promise.all([n.e(0),n.e(7)]).then(n.bind(null,"X7FI")).then(function(n){return t._isFirstParty(e.servicePrincipalId)?(t._firstPartyMsalTokenProvider||(t._firstPartyMsalTokenProvider=new n.MsalTokenProvider(e)),t._firstPartyMsalTokenProvider):(t._thirdPartyMsalTokenProvider||(t._thirdPartyMsalTokenProvider=new n.MsalTokenProvider(e)),t._thirdPartyMsalTokenProvider)})},e.prototype._getMsalBrowserTokenProvider=function(e){var t=this;return Object(c.t)()?Promise.all([n.e(12),n.e(4)]).then(n.bind(null,"oGcS")).then(function(n){return t._firstPartyMsalBrowserTokenProvider||(t._firstPartyMsalBrowserTokenProvider=new n.MsalBrowserTokenProvider(e)),t._firstPartyMsalBrowserTokenProvider}):Promise.all([n.e(13),n.e(5)]).then(n.bind(null,"6RFZ")).then(function(n){return t._firstPartyMsalBrowserTokenProvider||(t._firstPartyMsalBrowserTokenProvider=new n.MsalBrowserTokenProvider(e)),t._firstPartyMsalBrowserTokenProvider})},e.prototype._getImplicitTokenProvider=function(e){var t=this;return n.e(10).then(n.bind(null,"onrO")).then(function(n){return t._isFirstParty(e.servicePrincipalId)?(t._firstPartyImplicitTokenProvider||(t._firstPartyImplicitTokenProvider=new n.ImplicitFlowTokenProvider(e)),t._firstPartyImplicitTokenProvider):(t._thirdPartyImplicitTokenProvider||(t._thirdPartyImplicitTokenProvider=new n.ImplicitFlowTokenProvider(e)),t._thirdPartyImplicitTokenProvider)})},e.prototype._getOboTokenProvider=function(e,t,n,a){var i=this;return this._getAdalAuthContextManager(e).then(function(r){return i._isFirstParty(e.servicePrincipalId)?(i._firstPartyOboTokenProvider||(i._firstPartyOboTokenProvider=r.getOboTokenProvider(e,t,n,a)),i._firstPartyOboTokenProvider):(i._thirdPartyOboTokenProvider||(i._thirdPartyOboTokenProvider=r.getOboTokenProvider(e,t,n,a)),i._thirdPartyOboTokenProvider)})},e.prototype._isFirstParty=function(e){return e===s.e.PRE_AUTHORIZED_APP_PRINCIPAL_ID},e.prototype._shouldUseOboTokenExchange=function(){return(Object(d.d)()||(Object(c.a)()&&i._BrowserUtilities.isTeamsHosted()||!Object(d.l)()&&i._BrowserUtilities.isUsingSecureBroker()?i._BrowserUtilities.isTeamsHosted():i._BrowserUtilities.isWebViewHosted())||Object(c.r)()&&/.*AppleWebKit.*Safari/.test(navigator.userAgent))&&!!this._oboConfiguration},e}()},bB1B:function(e){e.exports=JSON.parse('{"e":"To view the information on this page, you need to verify your identity.","t":"To view the information on this page, ask a Global or SharePoint Administrator in your organization to go to the API management page in the new SharePoint admin center."}')}}]);
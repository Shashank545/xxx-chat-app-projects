(window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0=window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0||[]).push([[168],{ZSCv:function(e,t,n){"use strict";n.r(t),n.d(t,"supportsServiceWorker",function(){return l});var a,i=n("K9kD"),r=n("RJrw");!function(e){e[e.SharePoint=0]="SharePoint",e[e.Odc=1]="Odc",e[e.TeamsBroker=2]="TeamsBroker"}(a||(a={}));var o={FluentUIV9ShimOverallControl:60077,FluentUIV9ShimDefaultButton:60337,FluentUIV9ShimPrimaryButton:60344,FluentUIV9ShimIconButton:60355,FluentUIV9ShimActionButton:60356,FluentUIV9ShimImage:60357,FluentUIV9ShimToggle:60461,FluentUIV9ShimCommandBarButton:60466,FluentUIV9ShimCommandBar:60480,FluentUIV9ShimCommandButton:60501,FluentUIV9ShimTooltipHost:60634,FluentUIV9ShimTooltip:60633,FluentUIV9ShimLink:60405,FluentUIV9ShimStack:60413,FluentUIV9ShimCheckbox:60414,FluentUIV9ShimDialog:60426,FluentUIV9ShimChoiceGroup:60415,FluentUIV9ShimPersona:60419,FluentUIV9ShimFacepile:60569,FluentUIV9ShimPivot:60105,FluentUIV9ShimContextualMenu:60556,FluentUIV9ShimOverlay:60547,FluentUIV9ShimOverflowSet:60555,FluentUIV9ShimIcon:60446,FluentUIV9ShimSpinner:60450,FluentUIV9ShimSlider:60454,FluentUIV9ShimLayer:60460,FluentUIV9ShimText:60566,FluentUIV9ShimCustomizer:60650,FluentUIV9ShimThemeProvider:60567,FluentUIV9ShimSpinButton:60557,FluentUIV9ShimLabel:60382,FluentUIV9ShimSeparator:60384,FluentUIV9ShimCompoundButton:60540},s=[{registration:{id:"STS",swPrefetchManifestName:"stsserviceworkerprefetch"}},{registration:{id:"SPHome"}},{featureIds:[1966],registration:{id:"SitePages"}},{registration:{id:"Embed"}},{odc:!0,scenarioId:a.Odc,featureIds:["EnableODCServiceWorker"],registration:{id:"ODC"}},{featureIds:[60579],registration:{id:"CreateGroup"}},{featureIds:[60840],registration:{id:"TeamsLogon"}},{featureIds:[60155],registration:{id:"SingleWebPart"}},{featureIds:[60840],registration:{id:"VivaHome"}},{featureIds:[60877],registration:{id:"VivaAmplify"}},{featureIds:[60940],registration:{id:"BrokerLogon"}},{scenarioId:a.TeamsBroker,registration:{id:"TeamsBroker"}},{featureIds:[1344],registration:{id:"Clipchamp"}},{featureIds:[1988],registration:{id:"MeeBridge"}},{featureIds:[61128],registration:{id:"SPStart"}}];function c(e,t,n){var r=i.FeatureOverrides.isFeatureEnabled({ODC:!0}),c=r?a.Odc:"securebroker.sharepointonline.com"===location.host?a.TeamsBroker:a.SharePoint,d=s.filter(function(e){if(c!==(e.scenarioId||a.SharePoint))return!1;var t=!0;return e.featureIds&&e.featureIds.forEach(function(e){t=t&&i.FeatureOverrides.isFeatureEnabled(r?{ODC:e}:{ODB:e})}),"STS"===e.registration.id?t&&!navigator.userAgent.match(/Version\/[\d\.]+.*Safari/):t}).map(function(e){return e.registration}),l=JSON.stringify(d);n||(n=!r&&i.FeatureOverrides.isFeatureEnabled({ODB:1966})&&-1!==document.cookie.indexOf("SPClientSideFrameworkDevCookie-9076A2B0-2161-43C6-8BEA-9FECB7EA459B="));var u=i.FeatureOverrides.isFeatureEnabled({ODC:"DisablePickerSw"})||i.Killswitch.isActivated("EB4018E4-5975-493E-A50B-0BEA4CAFEA5A","05/10/2023","Disable ODC Picker usage of SW"),f=function(e){if(!i.Killswitch.isActivated("5FD21A1E-DDC7-4369-8264-307676C566A4","05/22/2023","Ability to temporarily disable custom nav preload")&&e===a.SharePoint&&i.FeatureOverrides.isFeatureEnabled({ODB:1855}))return{supportsFeatures:[1855]}}(c);return"".concat(r?"/odc":"/_layouts/15/odsp","serviceworkerproxy.aspx")+"?swManifestName=".concat(r?"odc":c===a.TeamsBroker?"teamsbroker":"sp","serviceworker")+"&debug=".concat(n?"true":"false")+"&bypass=".concat(t?"true":"false")+(f?"&navigationPreloadHeaderValue=".concat(encodeURIComponent(JSON.stringify(f))):"")+(e?"&userId=".concat(e):"")+"".concat(c===a.SharePoint&&i.FeatureOverrides.isFeatureEnabled({ODB:1235})?"&dataHost=Nucleus":"")+"&applications=".concat(encodeURIComponent(l))+"".concat(c===a.SharePoint&&i.FeatureOverrides.isFeatureEnabled({ODB:1855})?"&list=v2":"")+"".concat(c===a.SharePoint&&i.FeatureOverrides.isFeatureEnabled({ODB:60972})?"&listv2UseDataHostForResources=true":"")+"".concat(c===a.SharePoint&&i.FeatureOverrides.isFeatureEnabled({ODB:1878})&&HTMLScriptElement.supports&&HTMLScriptElement.supports("webbundle")?"&spHomeWebBundle=true":"")+"".concat(i.FeatureOverrides.isFeatureEnabled({ODB:60734})?"&defaultBrotli=true":"")+"".concat(i.FeatureOverrides.isFeatureEnabled({ODC:"EnableSwJsCaching"})?"&cacheResources=true":"")+"".concat(u?"&disablePickerSw=true":"")+"".concat(i.FeatureOverrides.isFeatureEnabled({ODB:60078})?"&authenticateFast=true":"")+"".concat(i.FeatureOverrides.isFeatureEnabled({ODB:60315})?"&prefetchFilebrowserPageInTeams=true":"")+"".concat(i.FeatureOverrides.isFeatureEnabled({ODB:60815})?"&".concat(function(){var e,t=new Array(1+(Object.keys(o).length>>5)).fill(0),n=0;for(e in o)if(o.hasOwnProperty(e)){var a=n>>5;i.FeatureOverrides.isFeatureEnabled({ODB:o[e]})&&(t[a]|=1<<(31&n)),n++}return"FUIV9Flights=[".concat(t.join(),"]")}()):"")}var d=function(){function e(){var e=this;this._listeners=new Set,this._onMessage=function(t){var n=t.data;e._listeners.forEach(function(e){e(n)})},l()&&navigator.serviceWorker.addEventListener("message",this._onMessage)}return e.prototype.addListener=function(e){this._listeners.add(e)},e.prototype.removeListener=function(e){this._listeners.delete(e)},e.prototype.register=function(e,t,n){if(!l())return Promise.reject();var a=c(e,t,n);return navigator.serviceWorker.register(a,{scope:"/",updateViaCache:!i.Killswitch.isActivated("229793C0-9B99-4ABB-80B1-DC8C5692682F","1/13/2023","updateViaCache")&&i.FeatureOverrides.isFeatureEnabled({ODB:1846})?"all":"none"})},e.prototype.registerAt=function(e,t,n,a){var i=this;return l()?Promise.resolve(e).catch(function(){}).then(function(e){if(null==e?void 0:e.length){var o=[],s=-1!==e.indexOf(location.hostname);s&&o.push(i.register().then());var d=s?e.length-1:e.length;o.push(d&&new Promise(function(e){var t=function(n){var a;n.data.name===r.e.ServiceWorkerRegistered&&(null===(a=document.getElementById("register-service-worker-".concat(n.data.host)))||void 0===a||a.remove(),--d||(removeEventListener("message",t),e()))};addEventListener("message",t)}));for(var l=0,u=e;l<u.length;l++){var f=u[l];if(!s||f!==location.hostname){var p=c(t,n,a),m="https://".concat(f).concat("/_layouts/15/RegisterServiceWorker.ashx").concat(new URL("https://".concat(f).concat(p)).search),_=document.createElement("iframe");_.name="register-service-worker",_.id="register-service-worker-".concat(f),_.src=m,_.hidden=!0,_.style.display="none",document.body.appendChild(_)}}return Promise.all(o).then()}}):Promise.reject()},e.prototype.unregister=function(){return l()?navigator.serviceWorker.getRegistrations().then(function(e){Promise.all(e.map(function(e){return e.unregister()}))}):Promise.reject()},e.prototype.send=function(e){if(l()&&navigator.serviceWorker.controller){var t=JSON.stringify(e);navigator.serviceWorker.controller.postMessage(t)}},e.prototype.sendAndReceiveReply=function(e){return l()&&navigator.serviceWorker.controller?new Promise(function(t,n){var a=new MessageChannel;a.port1.onmessage=function(e){var a=e.data;a.error?n(a.error):t(e.data)};var i=JSON.stringify(e);navigator.serviceWorker.controller.postMessage(i,[a.port2])}):Promise.reject(void 0)},e.prototype.sync=function(e){return l()?navigator.serviceWorker.ready.then(function(t){return t.sync.register(JSON.stringify(e))}):Promise.reject()},e.prototype.dispose=function(){l()&&navigator.serviceWorker.removeEventListener("message",this._onMessage)},e}();function l(){return"serviceWorker"in navigator}t.default=d}}]);
(window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0=window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0||[]).push([[160],{gghr:function(e,t,n){"use strict";n.r(t),n.d(t,"RecommendedItemsWrapper",function(){return p});var a=n("17wl"),i=n("cDcd"),r=n("UWqr"),o=n("ut3N"),s=n("2q6Q"),c=n("I6O9"),d=n("xs0R"),l=n("PJr9");function u(){return"IntersectionObserver"in window&&"IntersectionObserverEntry"in window&&"intersectionRatio"in IntersectionObserverEntry.prototype?("isIntersecting"in IntersectionObserverEntry.prototype||Object.defineProperty(IntersectionObserverEntry.prototype,"isIntersecting",{get:function(){return this.intersectionRatio>0}}),Promise.resolve()):n.e(101).then(n.t.bind(null,"r+er",7))}var f=n("7plR"),p=function(e){function t(n){var a=e.call(this,n)||this;a._loadPage=function(){if(!a._alreadyLoadingPage)try{a._alreadyLoadingPage=!0,a._deferredRenderStart=s._PerformanceLogger.now(),(!a._pageLoadMonitor||a._pageLoadMonitor&&a._pageLoadMonitor.hasEnded)&&(a._pageLoadMonitor=new o._QosMonitor("RecommendedItems.PageLoad")),s._EngagementLogger.logEventWithLogEntry(new o._LogEntry("RecommendedItems","PageLoad",o._LogType.Event,{referredRanking:a._referredRanking,isEnabled:a.props.isEnabled.toString()}));var e=a._fetchRecommendedItems();a._initWebpartObserver(e),a._loadComponent(),d.t.mode===r.DisplayMode.Read&&a.props.isEnabled&&e.then(function(e){e&&s._EngagementLogger.logEventWithLogEntry(new o._LogEntry("RecommendedItems","Load",o._LogType.Event,{referredRanking:a._referredRanking}))})}catch(e){a._logFailure(e)}finally{setTimeout(function(){a._alreadyLoadingPage=!1},1e3)}},a._ensureLatestEditingState=function(){var e=d.t.sliceUpdated;if(e&&e.indexOf(l.e.DisplayModeStore)>-1){var t=d.t.mode===r.DisplayMode.Edit;a.state.editing!==t&&a.setState({editing:t})}},a._ifRenderPhaseChanged=function(){return d.t.sliceUpdated&&d.t.sliceUpdated.indexOf(l.e.RenderPhaseStore)>-1},a._ifDisplayModeChanged=function(){return d.t.sliceUpdated&&d.t.sliceUpdated.indexOf(l.e.DisplayModeStore)>-1},a._ifNewPage=function(){var e=d.t.sliceUpdated;return e&&e.indexOf(l.e.RouteStore)>-1&&d.t.didTransition||!a.state.dataReturned&&!a.state.recommendedItemsClass&&d.t.didTransition&&!d.t.isTransitioning},a._onStoreChange=function(){a._deferredRenderMonitor.hasEnded&&(a._deferredRenderMonitor=new o._QosMonitor("RecommendedItems.DeferredRender")),a._ensureLatestEditingState(),a._ifRenderPhaseChanged()||a._ifDisplayModeChanged()?d.t.renderDeferredComponents&&a._loadPage():a._ifNewPage()&&(a.setState({dataReturned:!1}),a._loadPage())},a._getListId=function(){var e=f.e.instance.pageContext.list;return e?e.id:void 0},a._getUniqueId=function(){return d.t.fields?d.t.fields.UniqueId:void 0},a._doesUserHaveEditPermission=function(){return!d.t.fields||d.t.fields.DoesUserHaveEditPermission},a._fetchRecommendedItems=function(){var e={count:8,siteId:f.e.instance.pageContext.site.id,listId:a._getListId(),webId:f.e.instance.pageContext.web.id,uniqueId:a._getUniqueId(),qosMonitor:a._deferredRenderMonitor};return void 0===e.uniqueId||void 0===e.listId||!1===a.props.isEnabled&&!1===a._doesUserHaveEditPermission()?(a._alreadyLoadingPage=!1,void a.setState({dataReturned:!1,hasError:!0})):t.loadModule().then(function(e){return new e.RecommendedItemsDataProvider({serviceScope:f.e.instance.serviceScope})}).then(function(t){return t.requestData(e)}).then(function(e){return e.length>0?(a._preCachedItems=e,a._dataReturnedTime=s._PerformanceLogger.now(),a.setState({dataReturned:!0,hasError:!1}),{correlationId:e[0].correlationId,rankingAlgo:e[0].rankingAlgo}):(a._webPartObserver&&a._webPartObserver.disconnect(),void a.setState({dataReturned:!1,hasError:!0}))})},a._showOrHideRecommendedItems=function(e){var t=document.getElementById("RecommendedItems"),n=e?"448px":"1px";t&&(t.style.display=e?"flex":"none",t.style.minHeight=n)},a._deferredRenderMonitor=new o._QosMonitor("RecommendedItems.DeferredRender"),a._startTime=s._PerformanceLogger.now();var i=window.location?new URL(window.location.href):void 0;return a._referredRanking=i&&i.searchParams.get("from")||"",a._unsubscribe=d.t.addListener(a._onStoreChange),a.state={dataReturned:!1,editing:r.DisplayMode.Edit===d.t.mode,renderDeferred:!1},a}return Object(a.__extends)(t,e),t.loadModule=function(){return t._recItemsLibPromise||(t._recItemsLibPromise=c.SPComponentLoader.loadComponentById("279431a4-68f1-402c-ab81-ad5442f14179")),t._recItemsLibPromise},t.prototype.componentDidUpdate=function(){var e=this.state.hasError;this._rendered&&!e&&this._logSuccess("updated")},t.prototype.componentDidMount=function(){this.forceUpdate(),this._loadPage()},t.prototype.componentWillUnmount=function(){this._unsubscribe&&this._unsubscribe(),this.state.hasError||(this._rendered?this._logSuccess("unmounted"):this._rendered||this._logNoRender())},t.prototype.componentDidCatch=function(e){this._logFailure(e),this.setState({hasError:!0})},t.prototype.render=function(){var e=this.state,t=e.dataReturned,n=e.hasError,a=e.recommendedItemsClass,o=this.props.isEnabled,c=!n&&(o||d.t.mode===r.DisplayMode.Edit);if(this._showOrHideRecommendedItems(c),!c)return!1;if(a&&t){var l={dataResponse:this._preCachedItems,hideCallback:this._showOrHideRecommendedItems},u=i.createElement(a,l);return this._rendered=!0,this._renderTime=s._PerformanceLogger.now(),u}return i.createElement("div",{style:{height:"448px"}})},t.prototype._getLogExtraData=function(e){var t=this._startTime||0,n=this.state,a=n.dataReturned,i=n.recommendedItemsClass;return{initTime:t,renderTime:(this._deferredRenderStart||0)-t,moduleLoadTime:(this._dataReturnedTime||0)-t,duration:(this._renderTime||0)-t,dataProvider:!!a,loader:!!i,scenarioId:e||""}},t.prototype._logNoRender=function(){this._deferredRenderMonitor.hasEnded||this._deferredRenderMonitor.writeExpectedFailure("DidNotRender",void 0,this._getLogExtraData()),this._pageLoadMonitor&&!this._pageLoadMonitor.hasEnded&&this._pageLoadMonitor.writeExpectedFailure("DidNotRender")},t.prototype._logFailure=function(e){this._deferredRenderMonitor.hasEnded||this._deferredRenderMonitor.writeUnexpectedFailure("RenderFailed",e),this._pageLoadMonitor&&!this._pageLoadMonitor.hasEnded&&this._pageLoadMonitor.writeUnexpectedFailure("RenderFailed",e)},t.prototype._logSuccess=function(e){this._deferredRenderMonitor.hasEnded||(this._deferredRenderMonitor.writeSuccess(this._getLogExtraData(e)),this._pageLoadMonitor&&!this._pageLoadMonitor.hasEnded&&this._pageLoadMonitor.writeSuccess())},t.prototype._loadComponent=function(){var e=this;u().then(function(){return n.e(120).then(n.bind(null,"g/up"))}).then(function(t){return e.setState({recommendedItemsClass:t.default})})},t.prototype._initWebpartObserver=function(e){var t=this;u().then(function(){var n=t._getUniqueId();n!==t._uniqueId&&(t._uniqueId=n,t._hasUserScrolledDown=!1),t._webPartObserver=new IntersectionObserver(function(n,a){n.forEach(function(n){d.t.renderDeferredComponents&&n.intersectionRatio>0&&(t._hasUserScrolledDown||(a.disconnect(),d.t.mode===r.DisplayMode.Read&&t.props.isEnabled&&s._EngagementLogger.logEventWithLogEntry(new o._LogEntry("RecommendedItems","ScrolledDown",o._LogType.Event,{referredRanking:t._referredRanking})),e&&e.then(function(e){e&&d.t.mode===r.DisplayMode.Read&&t.props.isEnabled&&s._EngagementLogger.logEventWithLogEntry(new o._LogEntry("RecommendedItems","Seen",o._LogType.Event,{referredRanking:t._referredRanking,correlationId:e.correlationId,rankingAlgo:e.rankingAlgo}))}),t._hasUserScrolledDown=!0))})},{rootMargin:"0px",threshold:.1});var a=document.getElementById("RecommendedItems");a&&t._webPartObserver.observe(a)})},t}(i.Component)}}]);
(window.webpackJsonp_2e09fb9b_13bb_48f2_859f_97d6fff71176_1_3_325=window.webpackJsonp_2e09fb9b_13bb_48f2_859f_97d6fff71176_1_3_325||[]).push([[2],{idrx:function(e,t,n){"use strict";n.d(t,"t",function(){return l}),n.d(t,"e",function(){return u});var a=n("17wl"),i=0,r=["img","script","iframe","link","audio","video","source"];function o(e,t){for(var n=0,a=e;n<a.length;n++){var i=a[n];if(t.includes(i.nodeName.toLowerCase())||o(i.children,t))return!0}return!1}function s(){for(var e=[],t=0;t<arguments.length;t++)e[t]=arguments[t];c()&&console.log.apply(console,e)}function c(){try{if("sessionStorage"in window&&window.sessionStorage){var e=window.sessionStorage.enableTTILogging;return e&&"true"===e.toLowerCase()&&"undefined"!=typeof console&&!!console}}catch(e){}return!1}function d(e,t){if(e.length>2)return performance.now();for(var n=[],a=0,i=t;a<i.length;a++){var r=i[a];n.push({timestamp:r.start,type:"requestStart"}),n.push({timestamp:r.end,type:"requestEnd"})}for(var o=0,s=e;o<s.length;o++){var c=s[o];n.push({timestamp:c,type:"requestStart"})}n.sort(function(e,t){return e.timestamp-t.timestamp});for(var d=e.length,l=n.length-1;l>=0;l--){var u=n[l];switch(u.type){case"requestStart":d--;break;case"requestEnd":if(++d>2)return u.timestamp;break;default:throw Error("Internal Error: This should never happen")}}return 0}function l(){window.__tti&&window.__tti.o.observe({entryTypes:["longtask"]})}var u=function(){function e(e){void 0===e&&(e={});var t=this;this._useMutationObserver=!!e.useMutationObserver,this._trackNetworkRequests=!!e.trackNetworkRequests,this._minValue=e.minValue||null,this._requiredMainThreadQuietnessDurationInMilliseconds=e.requiredMainThreadQuietnessDurationInMilliseconds||5e3,this._requiredNetworkQuietnessDurationInMilliseconds=e.requiredNetworkQuietnessDurationInMilliseconds||5e3;var n=window.__tti&&window.__tti.e,a=window.__tti&&window.__tti.o;this._longTaskId=0,n?(s("Consuming the long task entries already recorded."),this._longTasks=n.map(function(e){return t._createLongTask(e)}),window.__tti.e=[]):this._longTasks=[],a&&a.disconnect(),this._networkRequests=[],this._incompleteJSInitiatedRequestStartTimes=new Map,this._timerId=void 0,this._timerActivationTime=-1/0,this._scheduleTimerTasks=!1,this._firstConsistentlyInteractiveResolver=null,this._performanceObserver=null,this._mutationObserver=null,this._registerListeners(this._trackNetworkRequests)}return e.prototype.getFirstConsistentlyInteractive=function(){var e=this;return new Promise(function(t,n){e._firstConsistentlyInteractiveResolver=t,"complete"===document.readyState?e.startSchedulingTimerTasks():window.addEventListener("load",function(){e.startSchedulingTimerTasks()})})},e.prototype.startSchedulingTimerTasks=function(){s("Enabling FirstConsistentlyInteractiveDetector"),this._scheduleTimerTasks=!0;var e=this._longTasks.length>0?this._longTasks[this._longTasks.length-1].end:0,t=d(this._incompleteRequestStarts(),this._networkRequests);this._rescheduleTimer(Math.max(this._trackNetworkRequests?t+this._requiredNetworkQuietnessDurationInMilliseconds:0,e))},e.prototype._rescheduleTimer=function(e){var t=this;this._scheduleTimerTasks?(s("Attempting to reschedule FirstConsistentlyInteractive "+"check to ".concat(e)),s("Previous timer activation time: ".concat(this._timerActivationTime)),this._timerActivationTime>e?s("Current activation time is greater than attempted reschedule time. No need to postpone."):(clearTimeout(this._timerId),this._timerId=setTimeout(function(){t._checkTTI()},Math.max(0,e-performance.now())),this._timerActivationTime=e,s("Rescheduled firstConsistentlyInteractive check at ".concat(e)))):s("startSchedulingTimerTasks must be called before calling rescheduleTimer")},e.prototype._disable=function(){s("Disabling FirstConsistentlyInteractiveDetector"),clearTimeout(this._timerId),this._scheduleTimerTasks=!1,this._unregisterListeners()},e.prototype._registerPerformanceObserver=function(e){var t=this;void 0===e&&(e=!0),this._performanceObserver=new PerformanceObserver(function(n){for(var a=0,i=n.getEntries();a<i.length;a++){var r=i[a];"resource"===r.entryType&&e&&t._networkRequestFinishedCallback(r),"longtask"===r.entryType&&t._longTaskFinishedCallback(r)}}),this._performanceObserver.observe({entryTypes:e?["longtask","resource"]:["longtask"]})},e.prototype._registerListeners=function(e){var t,n,a,s,c,d,l,u,f;void 0===e&&(e=!0),e&&(s=this._beforeJSInitiatedRequestCallback.bind(this),c=this._afterJSInitiatedRequestCallback.bind(this),d=XMLHttpRequest.prototype.send,l=i++,XMLHttpRequest.prototype.send=function(){for(var e=this,t=[],n=0;n<arguments.length;n++)t[n]=arguments[n];return s(l.toString()),this.addEventListener("readystatechange",function(){4===e.readyState&&c(l.toString())}),d.apply(this,t)},t=this._beforeJSInitiatedRequestCallback.bind(this),n=this._afterJSInitiatedRequestCallback.bind(this),a=window.fetch,window.fetch=function(){for(var e=[],r=0;r<arguments.length;r++)e[r]=arguments[r];return new Promise(function(r,o){var s=i++;t(s.toString()),a.apply(void 0,e).then(function(e){n(s.toString()),r(e)},function(e){n(e),o(e)})})}),this._registerPerformanceObserver(e),this._useMutationObserver&&(this._mutationObserver=(u=this._mutationObserverCallback.bind(this),(f=new MutationObserver(function(e){for(var t=0,n=e;t<n.length;t++){var a=n[t];("childList"===a.type&&o(a.addedNodes,r)||"attributes"===a.type&&r.includes(a.target.tagName.toLowerCase()))&&u(a)}})).observe(document,{attributes:!0,childList:!0,subtree:!0,attributeFilter:["href","src"]}),f))},e.prototype._unregisterListeners=function(){this._performanceObserver&&this._performanceObserver.disconnect(),this._mutationObserver&&this._mutationObserver.disconnect()},e.prototype._beforeJSInitiatedRequestCallback=function(e){s("Starting JS initiated request. Request ID: ".concat(e)),this._incompleteJSInitiatedRequestStartTimes.set(e,performance.now()),s("Active XHRs: ".concat(this._incompleteJSInitiatedRequestStartTimes.size))},e.prototype._afterJSInitiatedRequestCallback=function(e){s("Completed JS initiated request with request ID: ".concat(e)),this._incompleteJSInitiatedRequestStartTimes.delete(e),s("Active XHRs: ".concat(this._incompleteJSInitiatedRequestStartTimes.size))},e.prototype._networkRequestFinishedCallback=function(e){s("Network request finished",e),this._networkRequests.push({start:e.fetchStart,end:e.responseEnd}),this._rescheduleTimer(d(this._incompleteRequestStarts(),this._networkRequests)+this._requiredNetworkQuietnessDurationInMilliseconds)},e.prototype._longTaskFinishedCallback=function(e){s("Long task finished",e);var t,n=this._createLongTask(e),a="longTaskEnd".concat(n.id);t=a,window.performance&&performance.mark&&performance.mark(t),n.endMarkName=a,this._longTasks.push(n),this._rescheduleTimer(n.end+this._requiredMainThreadQuietnessDurationInMilliseconds)},e.prototype._mutationObserverCallback=function(e){s("Potentially network resource fetching mutation detected",e),s("Pushing back FirstConsistentlyInteractive check by ".concat(this._requiredNetworkQuietnessDurationInMilliseconds," milliseconds.")),this._rescheduleTimer(performance.now()+this._requiredNetworkQuietnessDurationInMilliseconds)},e.prototype._getMinValue=function(){return this._minValue?this._minValue:performance.timing.domContentLoadedEventEnd?performance.timing.domContentLoadedEventEnd-performance.timing.navigationStart:null},e.prototype._createLongTask=function(e){var t=e.startTime+e.duration;return{attribution:e.attribution,name:e.name,start:Math.round(e.startTime),end:Math.round(t),id:this._longTaskId++}},e.prototype._incompleteRequestStarts=function(){return Object(a.__spreadArray)([],Array.from(this._incompleteJSInitiatedRequestStartTimes.values()),!0)},e.prototype._checkTTI=function(){s("Checking if First Consistently Interactive was reached...");var e=performance.timing.navigationStart,t=d(this._incompleteRequestStarts(),this._networkRequests),n=(window.chrome&&window.chrome.loadTimes?1e3*window.chrome.loadTimes().firstPaintTime-e:0)||performance.timing.domContentLoadedEventEnd-e,a=this._getMinValue(),i=performance.now();if(null===a)return s("No usable minimum value yet. Postponing check."),void this._rescheduleTimer(Math.max(this._trackNetworkRequests?t+this._requiredNetworkQuietnessDurationInMilliseconds:0,i+1e3));s("Parameter values:"),s("NavigationStart",e),s("lastKnownNetwork2Busy",t),s("Search Start",n),s("Min Value",a),s("Last busy",t),s("Current time",i),s("Long tasks",this._longTasks),s("Incomplete JS Request Start Times",this._incompleteRequestStarts()),s("Network requests",this._networkRequests);var r=function(e,t,n,a,i,r,o,s){if(r&&a-n<s)return null;var c=0===i.length?e:i[i.length-1].end;if(a-c<o)return null;if(c&&i.length>0){var d=i[i.length-1].endMarkName;d&&performance.measure("TTI",void 0,d)}return Math.max(c,t)}(n,a,t,i,this._longTasks,this._trackNetworkRequests,this._requiredMainThreadQuietnessDurationInMilliseconds,this._requiredNetworkQuietnessDurationInMilliseconds);r&&this._firstConsistentlyInteractiveResolver?(s("maybeFCI",r),this._firstConsistentlyInteractiveResolver({tti:r,longTasks:this._longTasks}),this._disable()):(s("Could not detect First Consistently Interactive. Retrying in 1 second."),this._rescheduleTimer(performance.now()+1e3))},e}()},rC7z:function(e,t,n){"use strict";n.r(t),n.d(t,"getFirstConsistentlyInteractive",function(){return i});var a=n("idrx");function i(e){return void 0===e&&(e={}),"PerformanceLongTaskTiming"in window?new a.e(e).getFirstConsistentlyInteractive():Promise.resolve(null)}n.d(t,"restartLongTaskObserver",function(){return a.t})}}]);
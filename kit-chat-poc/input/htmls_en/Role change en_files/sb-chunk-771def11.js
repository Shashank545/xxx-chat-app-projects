(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["pro-kmfile"],{"+o28":function(e,t,r){"use strict";r.r(t),r.d(t,"registration",(function(){return b}));var n=r("+zb2"),a=r("UlRC"),i=r("rpSO"),c=r("KmCS"),o=r("F6lS"),u=r("x/lg"),l=r("a7o/"),s=r("Du1W"),f=r("05Fb"),d=r("HTw2"),p=r("V34+"),m=r("lS1Q");let h=class{constructor(e,t){this.topicScope=e,this.context=t}getId(){return"KMFileSuggestionProvider"}createStream(e){var t,r,n;const i=null===(t=this.context)||void 0===t?void 0:t.commanding.currentCommand.values[0].id;if(!i||!(null===(r=this.context)||void 0===r?void 0:r.instrumenter.props.serviceEnvInfo))return a.Observable.empty();const c=Object(d.a)(null===(n=this.context)||void 0===n?void 0:n.instrumenter.props.serviceEnvInfo);return Object(m.a)({topicId:i,environment:c,entity:"File",filterSpec:p.b,searchText:e}).select(({UniqueId:t,Title:r,Extension:n,Url:a})=>({entityType:"KMFile",id:t,displayTexts:[{text:Object(l.c)([e],r),icon:{fluentIcon:Object(f.a)(n,24),ariaLabel:Object(s.c)({ext:Object(u.a)({iconName:Object(f.a)(n,24)})})}}],url:a}))}};h=Object(n.__decorate)([Object(n.__param)(0,Object(i.a)("topicScope")),Object(n.__param)(1,Object(i.a)("context"))],h),t.default=h;const b=Object(c.regs)(`${o.a}.${h.prototype.getId()}`,h)},"3lF4":function(e,t,r){"use strict";var n=r("/krt"),a=r("pgot"),i=r("fhQA"),c=r("1zYj"),o=r("cDcd"),u=r("m+2y"),l=r("n8h4"),s=r("y1hr"),f=Object(l.b)((function(e){var t=Object(s.a)(e);return Object(u.b)({tidbit:t.tidbit,multiTidbit:t.multiTidbit,tidbitIcon:{padding:"0 2px 2px 0",verticalAlign:"middle"}})}));t.a=function(e){var t=e.tidbits,r=e.styles,c=e.theme,u=e.tidbitIcons,l=t.filter(Boolean),s=Object(a.a)()&&c?f(c):r;return l.length<=0?null:o.createElement("ul",{className:r.tidbits},l.map((function(e,t){var a,c;return o.createElement("li",{key:t,className:d(l,r,s)},u&&u[t]&&o.createElement(i.a,{iconName:null===(a=u[t])||void 0===a?void 0:a.icon,className:s.tidbitIcon,"aria-label":null===(c=u[t])||void 0===c?void 0:c.ariaLabel}),o.createElement(n.a,{highlightCssClass:r.highlighted,markType:"mark",text:e}),o.createElement("span",{className:r.offScreen},", "))})))};function d(e,t,r){return Object(a.a)()?1===e.length?null==r?void 0:r.tidbit:null==r?void 0:r.multiTidbit:Object(c.a)(t.tidbit,1===e.length?t.single:"")}},"6UuR":function(e,t,r){"use strict";r.d(t,"a",(function(){return a}));var n=r("HSwa");function a(e){var t=e.answerParentClassName,r=e.searchText,a=e.searchBox,i=e.dispatchTelemetryClickHandler,c=e.navigateToSERP;return function(e){var o,u=e.target,l=Object(n.c)(e.currentTarget),s=l?"ArrowLeft":"ArrowRight",f=l?"ArrowRight":"ArrowLeft";e.key===s&&u.classList.contains(t)&&(e.preventDefault(),e.stopPropagation(),null===(o=e.currentTarget.querySelectorAll("[data-nav='true']")[0])||void 0===o||o.focus());if("ArrowDown"===e.key&&u.classList.contains(t))return e.preventDefault(),e.stopPropagation(),void e.currentTarget.offsetParent.querySelectorAll("[data-nav='true']")[e.currentTarget.querySelectorAll("[data-nav='true']").length+1].focus();e.key===f&&"A"===u.tagName&&u.hasAttribute("data-nav")&&(e.preventDefault(),e.stopPropagation(),e.currentTarget.focus()),"Enter"===e.key&&u.classList.contains(t)&&(r?(i&&i(),e.preventDefault(),e.currentTarget.blur(),null==a||a.submitSearch(r),null==a||a.setSearchText(r)):c&&c(e))}}},At10:function(e,t,r){"use strict";function n(e){return null!=e}r.d(t,"a",(function(){return n}))},JczL:function(e,t,r){"use strict";var n=Math.ceil,a=Math.max;var i=function(e,t,r,i){for(var c=-1,o=a(n((t-e)/(r||1)),0),u=Array(o);o--;)u[i?o:++c]=e,e+=r;return u},c=r("bQU+"),o=r("/E5c");var u=function(e){return function(t,r,n){return n&&"number"!=typeof n&&Object(c.a)(t,r,n)&&(r=n=void 0),t=Object(o.a)(t),void 0===r?(r=t,t=0):r=Object(o.a)(r),n=void 0===n?t<r?1:-1:Object(o.a)(n),i(t,r,n,e)}}();t.a=u},Qhd2:function(e,t,r){"use strict";r.r(t),r.d(t,"filterOutNull",(function(){return a}));var n=r("At10");function a(e){return e.filter(n.a)}},lJaw:function(e,t,r){"use strict";r.d(t,"c",(function(){return n})),r.d(t,"b",(function(){return a})),r.d(t,"a",(function(){return i}));var n=["aspx","htm","html","mhtml"],a=function(e){return n.indexOf(e)>-1};function i(e){if(e){var t=e.lastIndexOf(".");if(t>-1)return e.substr(t)}return""}},qArz:function(e,t,r){"use strict";r.d(t,"a",(function(){return i}));var n={black:{red:0,green:0,blue:0,alpha:1},transparent:{red:0,green:0,blue:0,alpha:0},white:{red:255,green:255,blue:255,alpha:1}};function a(e,t){if(void 0===t&&(t=!1),(e=e.toLowerCase())in n)return n[e];var r=e.match(/^var\((--\w+)(?:, (#[0-9a-f]{3,6}))?\)$/);if(r){var a=function(e){var t=document.querySelector(":root body");if(!t)return"";var r=getComputedStyle(t).getPropertyValue(e);return r||""}(r[1]);!a&&r[2]?e=r[2]:a&&(e=a)}var i=e.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/);if(i)return t?{alpha:parseInt(i[1],16)/255,red:parseInt(i[2],16),green:parseInt(i[3],16),blue:parseInt(i[4],16)}:{red:parseInt(i[1],16),green:parseInt(i[2],16),blue:parseInt(i[3],16),alpha:parseInt(i[4],16)/255};var c=e.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/);if(c)return{red:parseInt(c[1],16),green:parseInt(c[2],16),blue:parseInt(c[3],16),alpha:1};var o=e.match(/^#([0-9a-f])([0-9a-f])([0-9a-f])([0-9a-f])$/);if(o)return t?{alpha:parseInt("".concat(o[1]),16)/15,red:parseInt("".concat(o[2]).concat(o[2]),16),green:parseInt("".concat(o[3]).concat(o[3]),16),blue:parseInt("".concat(o[4]).concat(o[4]),16)}:{red:parseInt("".concat(o[1]).concat(o[1]),16),green:parseInt("".concat(o[2]).concat(o[2]),16),blue:parseInt("".concat(o[3]).concat(o[3]),16),alpha:parseInt("".concat(o[4]),16)/15};var u=e.match(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/);if(u)return{red:parseInt("".concat(u[1]).concat(u[1]),16),green:parseInt("".concat(u[2]).concat(u[2]),16),blue:parseInt("".concat(u[3]).concat(u[3]),16),alpha:1};var l=e.replace(/\s/g,"").match(/^rgb\(([0-9\.]+),([0-9\.]+),([0-9\.]+)\)/);if(l)return{red:Number(l[1]),green:Number(l[2]),blue:Number(l[3]),alpha:1};var s=e.replace(/\s/g,"").match(/^rgba\(([0-9\.]+),([0-9\.]+),([0-9\.]+),([0-9\.]+)\)/);return s?{red:Number(s[1]),green:Number(s[2]),blue:Number(s[3]),alpha:Number(s[4])}:{red:0,green:0,blue:0,alpha:1}}function i(e,t,r){if(!t||!(t<0||t>1)){var n=a(e),i=n.red,c=n.green,o=n.blue,u=n.alpha,l=null!=t?t:u,s="".concat(i,",").concat(c,",").concat(o).concat(1!==l?",".concat(l):"");return r?s:1!==l?"rgba(".concat(s,")"):"rgb(".concat(s,")")}}},vbE8:function(e,t,r){"use strict";(function(e){var r=this&&this.__assign||function(){return(r=Object.assign||function(e){for(var t,r=1,n=arguments.length;r<n;r++)for(var a in t=arguments[r])Object.prototype.hasOwnProperty.call(t,a)&&(e[a]=t[a]);return e}).apply(this,arguments)};Object.defineProperty(t,"__esModule",{value:!0}),t.splitStyles=t.detokenize=t.clearStyles=t.loadTheme=t.flush=t.configureRunMode=t.configureLoadStyles=t.loadStyles=void 0;var n="undefined"==typeof window?e:window,a=n&&n.CSPSettings&&n.CSPSettings.nonce,i=function(){var e=n.__themeState__||{theme:void 0,lastStyleElement:void 0,registeredStyles:[]};e.runState||(e=r(r({},e),{perf:{count:0,duration:0},runState:{flushTimer:0,mode:0,buffer:[]}}));e.registeredThemableStyles||(e=r(r({},e),{registeredThemableStyles:[]}));return n.__themeState__=e,e}(),c=/[\'\"]\[theme:\s*(\w+)\s*(?:\,\s*default:\s*([\\"\']?[\.\,\(\)\#\-\s\w]*[\.\,\(\)\#\-\w][\"\']?))?\s*\][\'\"]/g,o=function(){return"undefined"!=typeof performance&&performance.now?performance.now():Date.now()};function u(e){var t=o();e();var r=o();i.perf.duration+=r-t}function l(){u((function(){var e=i.runState.buffer.slice();i.runState.buffer=[];var t=[].concat.apply([],e);t.length>0&&s(t)}))}function s(e,t){i.loadStyles?i.loadStyles(p(e).styleString,e):function(e){if("undefined"==typeof document)return;var t=document.getElementsByTagName("head")[0],r=document.createElement("style"),n=p(e),c=n.styleString,o=n.themable;r.setAttribute("data-load-themed-styles","true"),a&&r.setAttribute("nonce",a);r.appendChild(document.createTextNode(c)),i.perf.count++,t.appendChild(r);var u=document.createEvent("HTMLEvents");u.initEvent("styleinsert",!0,!1),u.args={newStyle:r},document.dispatchEvent(u);var l={styleElement:r,themableStyle:e};o?i.registeredThemableStyles.push(l):i.registeredStyles.push(l)}(e)}function f(e){void 0===e&&(e=3),3!==e&&2!==e||(d(i.registeredStyles),i.registeredStyles=[]),3!==e&&1!==e||(d(i.registeredThemableStyles),i.registeredThemableStyles=[])}function d(e){e.forEach((function(e){var t=e&&e.styleElement;t&&t.parentElement&&t.parentElement.removeChild(t)}))}function p(e){var t=i.theme,r=!1;return{styleString:(e||[]).map((function(e){var n=e.theme;if(n){r=!0;var a=t?t[n]:void 0,i=e.defaultValue||"inherit";return t&&!a&&console,a||i}return e.rawString})).join(""),themable:r}}function m(e){var t=[];if(e){for(var r=0,n=void 0;n=c.exec(e);){var a=n.index;a>r&&t.push({rawString:e.substring(r,a)}),t.push({theme:n[1],defaultValue:n[2]}),r=c.lastIndex}t.push({rawString:e.substring(r)})}return t}t.loadStyles=function(e,t){void 0===t&&(t=!1),u((function(){var r=Array.isArray(e)?e:m(e),n=i.runState,a=n.mode,c=n.buffer,o=n.flushTimer;t||1===a?(c.push(r),o||(i.runState.flushTimer=setTimeout((function(){i.runState.flushTimer=0,l()}),0))):s(r)}))},t.configureLoadStyles=function(e){i.loadStyles=e},t.configureRunMode=function(e){i.runState.mode=e},t.flush=l,t.loadTheme=function(e){i.theme=e,function(){if(i.theme){for(var e=[],t=0,r=i.registeredThemableStyles;t<r.length;t++){var n=r[t];e.push(n.themableStyle)}e.length>0&&(f(1),s([].concat.apply([],e)))}}()},t.clearStyles=f,t.detokenize=function(e){return e&&(e=p(m(e)).styleString),e},t.splitStyles=m}).call(this,r("qwMZ"))}}]);
(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["sug-univ"],{IIEc:function(e,t,n){"use strict";n.r(t);var r=n("6JqZ"),a=n("cDcd"),c=n("4zpW"),i=n("/Dog"),o=n("GtFt"),s=n("R0Os"),u=n("JnXn");const l=Object(s.a)(o.e,"endButton",(function(e,t={}){var n,r,c,o;const s=[e.empty||null===(n=e.styles)||void 0===n?void 0:n.clearButton,e.empty?null===(c=e.styles)||void 0===c?void 0:c.hidden:null===(r=e.styles)||void 0===r?void 0:r.visible].join(" ").trim(),l=e.searchBox;return a.createElement("button",{className:s,type:"button",title:e.label,"aria-label":e.label,onClick:()=>{var n;return null===(n=e.onAction)||void 0===n||n.call(e,p(t)),null==l||l.clear(),null==l?void 0:l.focus()},onMouseDown:u.b,"data-tab":!e.empty},a.createElement(i.a,{name:"icon_clear-x",wrapperClassName:null===(o=e.styles)||void 0===o?void 0:o.icon}))}));var d=l;const p=Object(c.a)(l,"onClick");var f=n("n5ha");const m=(Object(f.declareString)("search-box-container-plugins.searchux.strings.SearchBoxClearButton.label"));var h=n("QlCf"),v=n("n8h4"),g=n("m+2y");var b=Object(v.b)(e=>{const t=(e=>Object.assign(Object.assign({},Object(h.e)(e)),{backgroundColor:"transparent",padding:4}))(e),n=(e=>{const t=e.palette;return Object.assign(Object.assign({},h.f),{fontSize:e.fonts.medium.fontSize,color:t.neutralSecondary,fill:t.neutralSecondary,selectors:Object.assign(Object.assign({},Object(h.d)({color:"white",fill:"white"})),{"&:hover":{fill:t.themeDarkAlt,stroke:t.themeDarkAlt}})})})(e),r=h.b,a=h.g;return Object(g.b)({clearButton:t,icon:n,hidden:r,visible:a})});t.default=Object(r.a)(d,b,m)},LLyz:function(e,t,n){"use strict";function r(e,t,n){throw function(e,t,n){var r;return t&&t.safeToLog?(r=new Error("".concat(e," ").concat(t.message))).innerError=t:r=new Error(e),r.safeToLog=!0,r.logProperties=n,Object.freeze(r),r}(e,t,n)}n.d(t,"a",(function(){return r}))},dn8U:function(e,t,n){"use strict";n.r(t),n.d(t,"UniversalSuggestion",(function(){return x}));var r=n("+zb2"),a=n("NaTl"),c=n("wAGM"),i=n("ljXs"),o=n("Dxhy"),s=n("j4bS"),u=n("zt+T"),l=n("Icem"),d=n("cDcd"),p=n("YBZd"),f=n("gp5W"),m=n("pnoK"),h=n("X+5C"),v=n("m+2y"),g=n("n8h4"),b=n("y1hr"),y=Object(g.b)((function(e,t){var n=Object(b.a)(e),a=e.themeStyles.palette,c=e.componentStyles;return Object(v.b)(Object(r.__assign)(Object(r.__assign)({},n),{universalSuggestion:[n.suggestion,{padding:c.Suggestion.padding,height:t,selectors:{"&:hover":{cursor:"pointer"}}}],iconWrapper:[n.suggestionIconWrapper,{width:"24px",height:"24px",selectors:{"> img, > canvas":{verticalAlign:"middle"}}}],icon:{color:a.neutralPrimary,verticalAlign:"middle",fontSize:"16px",width:"24px !important",height:"24px !important"},rounded:{borderRadius:"50%"},squared:{borderRadius:"2px"}}))})),j=n("/krt"),O=n("3lF4");function I(e){var t=e.displayTexts,n=e.styles,a=e.theme,c=function(e){var t={icons:[],texts:[]};return e.slice(1).forEach((function(e){var n=e.icon,r=e.text;t.icons.push(n&&n.fluentIcon?{icon:n.fluentIcon,ariaLabel:n.ariaLabel}:void 0),t.texts.push(r)})),t}(t);return d.createElement("div",{className:n.suggestionTextWrapper},d.createElement("div",{className:n.suggestionTitle,"data-tooltip":!0},d.createElement(j.a,{markType:"mark",text:t[0].text,highlightCssClass:n.highlighted}),d.createElement("span",{className:n.offScreen},",")),d.createElement(O.a,Object(r.__assign)({},{styles:n,theme:a,tidbits:c.texts,tidbitIcons:c.icons})))}var x=function(e){function t(){return null!==e&&e.apply(this,arguments)||this}return Object(r.__extends)(t,e),t.prototype.render=function(){var e,t,n,c,o,s,u=this,v=(e=this.props).displayTexts,g=e.theme,b=e.entityType;if(!g||0===v.length||!b)return null;var j,O=y(g,v.length>1?"48px":"38px"),x="UniversalSuggestionsQuickActions_".concat(Object(i.a)()),_=Object(m.a)()?{"aria-describedby":x}:{};"People"===this.props.entityType&&(j={instrumenter:this.props.instrumenter});var T=(t=v[0]).icon,S=t.rawText,C=t.text,E=Object(r.__assign)(Object(r.__assign)({},T),{ariaLabel:(null==T?void 0:T.ariaLabel)||S||C,acronymIcon:(null==T?void 0:T.acronymIcon)&&void 0!==T.acronymIcon.acronym?Object(r.__assign)(Object(r.__assign)({},null==T?void 0:T.acronymIcon),{shape:(null===(n=null==T?void 0:T.acronymIcon)||void 0===n?void 0:n.shape)||((null===(c=null==T?void 0:T.acronymIcon)||void 0===c?void 0:c.isRounded)?"round":"square")}):void 0});return d.createElement(h.a,{theme:g,displayText:v[0].rawText||l.a.getTextFromHtml(v[0].text)},d.createElement(a.a,null,d.createElement("a",Object(r.__assign)({href:this.props.url||"#",className:O.universalSuggestion,target:"_blank",rel:"noopener noreferrer",onClick:this.onClick},_),(null===(o=v[0])||void 0===o?void 0:o.icon)&&d.createElement(p.a,{icon:E,styles:O}),d.createElement(I,{displayTexts:v,styles:O,theme:g}),d.createElement(f.a,{entityType:b,onClick:function(e,t){var n,r;return null===(r=(n=u.props).dispatchImpressionClick)||void 0===r?void 0:r.call(n,e,{commandId:t.id})},entityId:this.props.id,theme:g,customProps:Object(r.__assign)(Object(r.__assign)(Object(r.__assign)({},this.props.customProps),j),{position:null===(s=this.props.logProps)||void 0===s?void 0:s.position,context:this.props.context,searchBox:this.props.searchBox}),describedById:x}))))},t.prototype.onClick=function(e){var t,n;null===(n=(t=this.props).onClick)||void 0===n||n.call(t,e),e.currentTarget.blur()},Object(r.__decorate)([u.a,Object(o.a)(),s.a],t.prototype,"onClick",null),t}(d.Component);t.default=Object(c.a)("UniversalSuggestion")(x)},k6ph:function(e,t,n){"use strict";n.d(t,"a",(function(){return I}));var r,a,c=n("+zb2"),i=n("2CHH"),o=n("3UHF"),s=!1,u=!1;function l(e,t,n,r){var a;f(n);var c=d(r);try{var o=c.getItem(e);if(!o)return null;a=JSON.parse(o)}catch(e){return!s&&Object(i.getDispatcher)().dispatch({eventType:"ERROR",name:"StorageCache",detail:"Failed to read from storage."}),s=!0,null}return a?a.cacheVersion!==t?(c.removeItem(e),null):a.puid&&a.puid===n?a:(c.removeItem(e),!!0&&Object(i.getDispatcher)().dispatch({eventType:"ERROR",name:"StorageCache",detail:"Invalid puid detected."}),null):null}var d=function(e){return e?(r||(r={getItem:function(e){return localStorage.getItem(p(e))},removeItem:function(e){return localStorage.removeItem(p(e))},setItem:function(e,t){localStorage.setItem(p(e),t)}}),r):(a||(a={getItem:function(e){return sessionStorage.getItem(p(e))},removeItem:function(e){return sessionStorage.removeItem(p(e))},setItem:function(e,t){sessionStorage.setItem(p(e),t)}}),a)};function p(e){return"mssearchux-cache-".concat(e)}function f(e){!e&&Object(o.a)("Puid not set.")}function m(e,t,n,r){var a=l(e,t,n,r);if(!a)return null;var c=a.cacheItems;return Object.keys(c).reduce((function(e,t){return e[t]={value:Promise.resolve({xhr:{responseText:c[t].value,status:c[t].status||0}}),expiryTime:c[t].expiryTime,cacheItemType:"storage",cacheItemResolved:!0},e}),{})}function h(e,t){return e.cacheItems[t]=null}function v(e,t,n,r,a){var o=t.cacheId,s=t.cacheVersion,p=t.cacheItemLifetime,m=t.useCachePersistentStorage;f(n);var v=e.cacheItems,b=g(v,a,{value:r,expiryTime:Date.now()+p,cacheItemType:"memory",cacheItemResolved:!1});return r.then((function(e){return g(v,a,Object(c.__assign)(Object(c.__assign)({},b),{cacheItemResolved:!0})),function(e,t,n,r,a,c,o,s){f(n);var p=l(e,t,n,s)||{cacheCreationTime:Date.now(),cacheVersion:t,puid:n,cacheItems:{}};p.cacheItems[r]={value:a,expiryTime:c,status:o};try{d(s).setItem(e,JSON.stringify(p))}catch(t){!u&&Object(i.getDispatcher)().dispatch({eventType:"ERROR",name:"StorageCache",nameDetail:e,detail:"Failed to write to session storage."}),u=!0}}(o,s,n,a,e.xhr.responseText,b.expiryTime,e.xhr.status,m),Object(c.__assign)(Object(c.__assign)({},e),{cacheItemResolved:!1,cacheItemType:null,expiryTime:b.expiryTime})}),(function(t){throw h(e,a),t}))}var g=function(e,t,n){return e[t]=n};function b(e,t){var n=t.cacheId,r=t.cacheVersion,a=t.cacheItemLifetime,i=t.userInfoPuid,o=t.useCachePersistentStorage;return{operation:function(t,s,u){var l=function(e,t,n,r,a){f(r);var c=e[t];return c&&c.puid!==r&&(c=null),c||(e[t]={puid:r,cacheItems:m(t,n,r,a)||{}})}(e,n,r,i,o);return"setItem"!==t&&function(e,t){var n=e.cacheItems[t];if(n){if(Date.now()<n.expiryTime)return n.value.then((function(e){return Object(c.__assign)(Object(c.__assign)({},e),{cacheItemType:n.cacheItemType,cacheItemResolved:n.cacheItemResolved,expiryTime:n.expiryTime})}));h(e,t)}return null}(l,s)||v(l,{cacheId:n,cacheVersion:r,cacheItemLifetime:a,useCachePersistentStorage:o},i,u(),s)}}}function y(e){return"object"==typeof e&&null!==e}var j=n("DeF7"),O=n("xn41");function I(e){return function(t){var n=t.cacheId,r=t.cacheVersion,a=t.cacheItemLifetime,c=t.cacheKeyExtractor,i=t.forceRefresh,o=t.useCachePersistentStorage;return function(t){var s=t.prefetchCacheCollection||_,u=function(e){if(!y(e))throw new Error("Key argument must be an object.");for(var t=Object.keys(e).sort(),n=t.length,r=[],a=0;a<n;a++){var c=t[a],i=e[c];void 0===i||y(i)||"function"==typeof i||r.push(c+":"+i)}return r.join(",")}(c(function(e){return Object.keys(e).reduce((function(t,n){return n.length>"prefetch".length&&0===n.indexOf("prefetch")||(t[n]=e[n]),t}),{})}(t))||{}),l=(null==i?void 0:i(t))?"setItem":"getItem";return s.bind({cacheConsumer:x,cacheId:n,cacheVersion:r,cacheItemLifetime:a,userInfoPuid:t.prefetchUserInfoPuid||Object(j.c)(),useCachePersistentStorage:o}).operation(l,u,(function(){return e(t)}))}}}var x={packageName:"@1js/search-prefetch",packageVersion:O.a},_={bind:function(e){return b(Object(j.a)().caches,e)}}},q9uV:function(e,t,n){"use strict";n.d(t,"a",(function(){return o})),n.d(t,"b",(function(){return s}));var r=n("UqTr"),a=n("ABrm"),c=n("00RH"),i=n("V7by");function o(e){return t=>(t.substrateSearchServiceProviderProps=e,t)}function s(e,t){const n=Object(r.a)(Object(a.a)(Object(c.a)(e.groups.map(e=>e.providers.map(u)))).map(({entityType:e})=>e));return(t||"").trim()?n:Object(i.a)(n,["Documents"])}function u(e){return e.substrateSearchServiceProviderProps}},uVNa:function(e,t,n){"use strict";var r=n("EhQd"),a=n("jfYR"),c=n("/QIa"),i=n("3AH6"),o=n("bWJB");var s=function(e){for(var t,n=[];!(t=e.next()).done;)n.push(t.value);return n},u=n("8gmL"),l=n("ohIh");var d=function(e){return e.split("")},p=n("f/ur"),f="[\\ud800-\\udfff]",m="[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]",h="\\ud83c[\\udffb-\\udfff]",v="[^\\ud800-\\udfff]",g="(?:\\ud83c[\\udde6-\\uddff]){2}",b="[\\ud800-\\udbff][\\udc00-\\udfff]",y="(?:"+m+"|"+h+")"+"?",j="[\\ufe0e\\ufe0f]?"+y+("(?:\\u200d(?:"+[v,g,b].join("|")+")[\\ufe0e\\ufe0f]?"+y+")*"),O="(?:"+[v+m+"?",m,g,b,f].join("|")+")",I=RegExp(h+"(?="+h+")|"+O+j,"g");var x=function(e){return e.match(I)||[]};var _=function(e){return Object(p.a)(e)?x(e):d(e)},T=n("tH0V"),S=r.a?r.a.iterator:void 0;t.a=function(e){if(!e)return[];if(Object(i.a)(e))return Object(o.a)(e)?_(e):Object(a.a)(e);if(S&&e[S])return s(e[S]());var t=Object(c.a)(e);return("[object Map]"==t?u.a:"[object Set]"==t?l.a:T.a)(e)}},x2tm:function(e,t,n){"use strict";n.d(t,"a",(function(){return i}));var r=n("+zb2");function a(e,t){var n={};if(e.getAllResponseHeaders){var r=e.getAllResponseHeaders();t.forEach((function(t){e.getResponseHeader&&-1!==r.indexOf(t)&&(n[t]=e.getResponseHeader(t)||"")}))}return n}var c=n("uKfc");function i(e,t){var n=e.monitorName,a=e.monitorType,i=void 0===a?"prefetch_request":a,s=e.additionalSuccessPropsExtractor,u=e.additionalFailurePropsExtractor;return Object(c.a)({monitorName:n,monitorType:i,successPropExtractor:function(e){var t=e.xhr,n=e.tokenAttemptCount,a=e.tokenFetchDuration,c=e.cacheItemResolved,i=e.cacheItemType;return Object(r.__assign)(Object(r.__assign)(Object(r.__assign)(Object(r.__assign)({},o(t)),n?{tokenAttemptCount:n,tokenFetchDuration:Math.round(a)}:{}),{cacheItemResolved:c,cacheItemType:i}),null==s?void 0:s(t))},failurePropExtractor:function(e){var t;return Object(r.__assign)(Object(r.__assign)(Object(r.__assign)({},null==u?void 0:u(e)),o(e)),{errorCode:null===(t=e.status)||void 0===t?void 0:t.toString()})}},(function(e){return void 0===e?Promise.reject("undefined options"):t(e)}))}function o(e){if(!e||!e.getAllResponseHeaders)return{};var t=a(e,["sprequestduration","spclientservicerequestduration","sprequestguid"]),n={};return n.sprequestguid=t.sprequestguid,n.sprequestduration=t.sprequestduration||t.spclientservicerequestduration||void 0,n}}}]);
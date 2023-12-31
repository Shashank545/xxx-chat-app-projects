(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["cmd-deletepqh"],{"3tkQ":function(e,t,r){"use strict";r.r(t);var n=r("n5ha"),o=r("nPTt"),i=r("WMEh"),a=function(e){return{authorization:"Bearer ".concat(e),"Content-Type":"application/json"}},s=Object(i.a)((function(e){return Object(o.a)((function(t){var r=t.queryText,n=t.clientType;return{url:"https://substrate.office.com/search/api/v1/searchhistory",method:"POST",body:JSON.stringify({Action:"SingleDelete",Scenario:{Name:n},DeletedQuery:r}),headers:a(e)}}))})),c=r("pDa8"),u=r("ohlq"),l=r("GPGK"),p=(Object(n.declareString)("search-dl.quickActions.deletePQH.tooltip")),m=(Object(n.declareString)("search-dl.quickActions.deletePQH.errorTooltip")),h=(Object(n.declareString)("search-dl.quickActions.deletePQH.loadingTooltip")),d=Object(l.a)({id:"DeletePQH",entityType:"Text",parameters:[{name:"text",entity:"Text",isRequired:!0,validate:function(e){var t,r,n,o;return!!((null===(t=e.customProps)||void 0===t?void 0:t.isHistory)||(null===(r=e.customProps)||void 0===r?void 0:r.position)||(null===(n=e.customProps)||void 0===n?void 0:n.queryText)||(null===(o=e.customProps)||void 0===o?void 0:o.searchBox))}}],execute:function(e){var t=e.text,r=t.value.customProps,n=r.searchBox,o=r.position,i=r.queryText,a=r.clearCache,l=t.toolTip;return s({queryText:i,clientType:Object(u.a)(),tokenProvider:Object(c.a)()}).then((function(){n.focus(),l.dismiss(),function(e){return document.querySelector('[data-suggestion-position="'.concat(e,'"]'))}(o).remove(),a()}))},additionalProps:{icon:{iconName:"Cancel"},tooltip:p,loadingTooltip:h,showLoading:!0,showError:!0,errorTooltip:m}});t.default=d},"6SpP":function(e,t,r){"use strict";var n,o=r("UlRC"),i=r("udHp"),a=r("7yNR"),s=r("lonS"),c=r("EH6A"),u=r("TxZW");"object"==typeof window&&(n=window.sessionStorage)||(n={getItem:function(){return null},removeItem:function(){},setItem:function(){},clear:function(){return{}}});var l=Object(a.a)((function(){return new p})),p=function(){function e(){this.cachedStreams={},this.sessionStorageError=!1}return e.prototype.clear=function(){this.cachedStreams={},this.sessionStorageError=!1},e.prototype.removeEntry=function(e){delete this.cachedStreams[e],n.removeItem(e),this.sessionStorageError=!1},e.currentTime=function(){return(new Date).getTime()},e.serializeKey=function(e){if(!Object(s.a)(e))throw new Error("Key argument must be an object.");for(var t,r,n=0,o=Object.keys(e).sort(),i=o.length,a=[];n<i;n++)null===(t=e[r=o[n]])||void 0===t||Object(s.a)(t)||Object(c.a)(t)||(t+="")&&a.push(r+":"+t);return a.join(",")},e.prototype.getEntryFromStreamCache=function(e,t){var r=this.cachedStreams[e];return r?r.version!==t?(delete this.cachedStreams[e],null):r:null},e.prototype.getEntryFromSessionStorage=function(t,r){var o;try{var i=n.getItem(t);if(!i)return null;o=JSON.parse(i)}catch(t){return this.sessionStorageError||(this.sessionStorageError=!0,Object(u.a)({eventType:"ERROR",name:e.sessionStorageErrorName,detail:"Failed to read from session storage."})),null}return o?o.version!==r?(n.removeItem(t),null):o:null},e.prototype.setSessionStorageItem=function(t,r){try{n.setItem(t,JSON.stringify(r))}catch(t){this.sessionStorageError||(this.sessionStorageError=!0,Object(u.a)({eventType:"ERROR",name:e.sessionStorageErrorName,detail:"Failed to write to session storage."}))}},e.prototype.saveToSessionStorage=function(e,t,r,n,o){var i=this.getEntryFromSessionStorage(e,r);i||(i={version:r,items:{}}),i.items[t]={value:n,expiryTime:o},this.setSessionStorageItem(e,i)},e.prototype.addStreamToCache=function(e,t,r,n,o,i){var a=this.getEntryFromStreamCache(e,r);a||(a={version:r,items:{}}),a.items[t]={value:n,expiryTime:o,streamState:i},this.cachedStreams[e]=a},e.prototype.removeStream=function(e,t){var r=this.cachedStreams[e];r&&delete r.items[t]},e.prototype.getStream=function(t,r,n){var i=e.serializeKey(r),a=this.getEntryFromStreamCache(t,n),s=a&&a.items[i];if(s)return e.currentTime()>s.expiryTime?(this.removeStream(t,i),null):{stream:s.value,streamState:s.streamState};var c=this.getEntryFromSessionStorage(t,n);if(c){var u=c.items[i];if(u){if(e.currentTime()>u.expiryTime)return delete c.items[i],this.setSessionStorageItem(t,c),null;var l=o.Observable.returnValue(u.value);return this.addStreamToCache(t,i,n,l,u.expiryTime,"Complete"),{stream:l,streamState:"Complete"}}}return null},e.prototype.saveStream=function(t,r,n,o,i,a){var s=this,c=e.serializeKey(r);this.addStreamToCache(t,c,n,o,Number.MAX_VALUE,"Pending"),o.subscribe((function(e){if(a&&a(e))s.removeStream(t,c);else{var r=Number.MAX_VALUE;void 0!==i&&(r=(new Date).getTime()+i),s.saveToSessionStorage(t,c,n,e,r);var o=s.cachedStreams[t];if(o){var u=o.items[c];u&&(o.items[c]={value:u.value,expiryTime:r,streamState:"Complete"})}}}),(function(){s.removeStream(t,c)}))},e}();r("O38G");function m(e,t,r,n,a,s){var c,u,p=null,m=o.Observable.create((function(r){var n,o=!p;p||(c=0,u=!1,p=new i.ReplaySubject),n=c<=1?"New":u?"Complete":"Pending";var a=p.subscribe((function(e){u=!0,r.onNext({value:e,streamState:n,clearCache:function(){return l().removeEntry(t)}})}),(function(e){return r.onError(e)}),(function(){return r.onCompleted()}));return o&&e.subscribe((function(e){p&&p.onNext(e)}),(function(e){p&&p.onError(e),p=null}),(function(){return p&&p.onCompleted()})),a})),h=m.select((function(e){return e.value}));return l().saveStream(t,r,n,h,a,s),m.onSubscribe((function(){c++}))}o.Observable.prototype.cache=function(e,t,r,n,i){var a=this,s=l().getStream(e,t,r);if(!s)return m(this,e,t,r,n,i);var c=!1;return o.Observable.create((function(o){if(c&&!(s=l().getStream(e,t,r)))return m(a,e,t,r,n,i);c=!0,s&&s.stream.subscribe((function(t){s&&o.onNext({value:t,streamState:s.streamState,clearCache:function(){return l().removeEntry(e)}})}),(function(e){return o.onError(e)}),(function(){return o.onCompleted()}))}))}},"9zDX":function(e,t,r){"use strict";r.d(t,"a",(function(){return c}));var n=/[\(\[\{\<][^\)\]\}\>]*[\)\]\}\>]/g,o=/[\0-\u001F\!-/:-@\[-`\{-\u00BF\u0250-\u036F\uD800-\uFFFF]/g,i=/^\d+[\d\s]*(:?ext|x|)\s*\d+$/i,a=/\s+/g,s=/[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\uD840-\uD869][\uDC00-\uDED6]/;function c(e,t,r){return e?(e=function(e){return e=(e=(e=(e=e.replace(n,"")).replace(o,"")).replace(a," ")).trim()}(e),s.test(e)||!r&&i.test(e)?"":function(e,t){var r="",n=e.split(" ");return 2===n.length?(r+=n[0].charAt(0).toUpperCase(),r+=n[1].charAt(0).toUpperCase()):3===n.length?(r+=n[0].charAt(0).toUpperCase(),r+=n[2].charAt(0).toUpperCase()):0!==n.length&&(r+=n[0].charAt(0).toUpperCase()),t&&r.length>1?r.charAt(1)+r.charAt(0):r}(e,t)):""}},P6pV:function(e,t,r){"use strict";r.r(t);var n=r("+zb2"),o=r("/krt"),i=r("NaTl"),a=r("wAGM"),s=r("Dxhy"),c=r("j4bS"),u=r("zt+T"),l=r("Icem"),p=r("cDcd"),m=r("X+5C"),h=r("PS6g"),d=r("m+2y"),g=r("n8h4"),f=r("y1hr"),v=Object(g.b)((function(e,t){var r,o;void 0===e&&(e=h.b),void 0===t&&(t=0);var i=Object(f.a)(e),a=e.themeStyles.palette;return Object(d.b)(Object(n.__assign)(Object(n.__assign)({suggestionContainer:{selectors:{"&:hover":{background:"transparent !important"}}},iconWrapper:[i.suggestionIconWrapper,(r={display:"flex",fontSize:"28px",lineHeight:"inherit",minHeight:"52px",height:"52px",width:"52px",maxWidth:"52px",boxSizing:"border-box",border:"1px solid ".concat(e.isDarkMode?a.neutralPrimaryAlt:a.neutralQuaternaryAlt),borderRadius:"6px"},r[".".concat(f.d,":hover &")]={background:a.neutralLighter,borderColor:e.isDarkMode?a.neutralSecondaryAlt:a.neutralQuaternaryAlt},r)],icon:{backgroundColor:"rgba(0, 0, 0, 0)",width:"28px",height:"28px"},displayText:[i.suggestionTitle,Object(n.__assign)(Object(n.__assign)({},t>10&&{display:"-webkit-box","-webkit-box-orient":"vertical","-webkit-line-clamp":"2",wordWrap:"break-word"}),(o={fontSize:"12px",color:a.neutralSecondary,lineHeight:"16px",textOverflow:"ellipsis",overflow:"hidden",whiteSpace:"initial"},o[".".concat(f.d,":hover &")]={fontWeight:600},o))],secondaryDisplayText:i.suggestionSecondaryTitle},i),{textWrapper:[i.suggestionTextWrapper,{padding:"4px",maxWidth:"80%",textAlign:"center"}],suggestion:[i.suggestion,{height:"auto !important",flexDirection:"column",justifyContent:"center",alignItems:"center",paddingRight:"1px",paddingTop:"7px",selectors:{":hover":{backgroundColor:"initial !important"}}}]}))}),1),y=function(e){function t(){return null!==e&&e.apply(this,arguments)||this}return Object(n.__extends)(t,e),t.prototype.render=function(){var e,t=v(this.props.theme,null===(e=this.props.title)||void 0===e?void 0:e.length);return p.createElement(m.a,{theme:this.props.theme,displayText:this.props.displayTextRaw||l.a.getTextFromHtml(this.props.displayText),className:t.suggestionContainer},p.createElement(i.a,null,p.createElement("a",{href:this.props.url,target:"_blank",onClick:this.onClick,className:t.suggestion,rel:"noreferrer"},p.createElement("div",{className:"".concat(t.iconWrapper," o365cs-base")},this.props.iconClass?p.createElement("div",{style:{color:this.props.color,margin:"auto"},className:"".concat(this.props.iconClass,"\n              ms-fcl-tp\n              ").concat(t.icon)}):p.createElement("img",{className:t.icon,src:this.props.iconUrl,alt:""})),p.createElement("div",{className:t.textWrapper},p.createElement("div",{className:t.displayText,"data-tooltip":!0},p.createElement(o.a,{markType:"mark",text:this.props.title,highlightCssClass:t.highlighted})),p.createElement("div",{className:t.secondaryDisplayText},this.props.secondaryDisplayText)))))},t.prototype.onClick=function(e){var t,r;null===(r=(t=this.props).onClick)||void 0===r||r.call(t,e),e.currentTarget.blur()},t.defaultProps={onClick:function(){return null}},Object(n.__decorate)([u.a,Object(s.a)(),c.a],t.prototype,"onClick",null),t=Object(n.__decorate)([Object(a.a)("HorizontalAppSuggestion")],t)}(p.Component);t.default=y},XgOi:function(e,t,r){"use strict";r.d(t,"a",(function(){return o}));var n=r("a7o/"),o=function(){function e(){this.searchSpec={title:n.d()},this.highlightSpec={title:n.c}}return e.prototype.scorePredicate=function(e){return e.title},e}()},kCs8:function(e,t,r){"use strict";var n=r("pinr"),o=r("rbtW");var i=function(e,t){return Object(n.a)(e,t,(function(t,r){return Object(o.a)(e,r)}))},a=r("00RH"),s=r("V47J"),c=r("mcBO");var u=function(e){return Object(c.a)(Object(s.a)(e,void 0,a.a),e+"")}((function(e,t){return null==e?{}:i(e,t)}));t.a=u}}]);
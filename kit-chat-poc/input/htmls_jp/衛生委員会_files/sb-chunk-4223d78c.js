(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["cmd-copyfileurl"],{"1zYj":function(e,r,n){"use strict";function t(){for(var e=[],r=0;r<arguments.length;r++)e[r]=arguments[r];for(var n=[],t=0,o=e;t<o.length;t++){var a=o[t];if(a)if("string"==typeof a)n.push(a);else if(a.hasOwnProperty("toString")&&"function"==typeof a.toString)n.push(a.toString());else for(var i in a)a[i]&&n.push(i)}return n.join(" ")}n.d(r,"a",(function(){return t}))},"3Ir8":function(e,r,n){"use strict";n.r(r);var t=n("9YZs"),o=n("9LXr"),a=n("UZl3"),i=n("GPGK"),c=Object(i.a)({id:"CopyLink",entityType:"File",parameters:[{name:"file",entity:"File",isRequired:!0,validate:function(e){var r;return e.id.startsWith("SPO_")&&!!Object(o.a)((null===(r=e.customProps)||void 0===r?void 0:r.fileUrl)||e.id)}}],execute:function(e){var r=e.file.value,n=r.customProps,o=r.id;return Object(t.a)(o,(null==n?void 0:n.fileUrl)||o,!0)},additionalProps:{tooltip:a.a,icon:{iconName:"Link"}}});r.default=c},"7w/n":function(e,r,n){"use strict";n.r(r),n.d(r,"registration",(function(){return h}));var t=n("+zb2"),o=n("rpSO"),a=n("KmCS"),i=n("F6lS"),c=n("POWj"),s=n("bs+K"),u=n("a7o/"),l=n("ohlq"),d=n("xo1z"),p=n("qGDX"),g=n("q9uV");let f=class{constructor(e,r,n,t){this.config=e,this.fetcher=r,this.site=n,this.conversation=t}getId(){return"AutoSuggestSiteScopedSiteSuggestionProvider"}createStream(e){return Object(p.b)(this.fetcher,"Site",{searchText:e,entityTypes:Object(g.b)(this.config,e),scenario:Object(d.a)(Object(l.a)()),appOverride:Object(c.a)(),logicalSearchId:this.conversation&&this.conversation.getLogicalSearchId("sb"),conversationId:this.conversation&&this.conversation.conversationId}).select(({AccessUrl:r,Acronym:n,Color:t,Title:o,logProps:a,originalLogicalSearchId:i,entityType:c})=>({id:r,acronym:n,color:t,searchText:e,title:Object(u.c)([e],Object(s.a)(o)),url:r,secondaryDisplayText:r,displayText:Object(u.c)([e],Object(s.a)(o)),logProps:a,originalLogicalSearchId:i,entityType:c}))}};f=Object(t.__decorate)([Object(g.a)({entityType:"Site"}),Object(t.__param)(0,Object(o.a)("config")),Object(t.__param)(1,Object(o.a)("fetcher")),Object(t.__param)(2,Object(o.a)("site")),Object(t.__param)(3,Object(o.a)("conversation"))],f),r.default=f;const h=Object(a.regs)(i.a+".autosuggestsitescopedsite",f)},H3jn:function(e,r,n){"use strict";n.d(r,"a",(function(){return s}));var t=n("UlRC"),o=n("ABrm"),a=n("Ta5s"),i=n("QxCF"),c=n("oo+0");function s(e,r,n){if(!e)throw new Error("Search box configuration must be specified.");const{streamPolicy:s,groups:l}=e,d=l.map(t=>new c.a(e,t,r).createStream(n)),p=d.map(e=>e.slice(0,e.length-1)),g=t.Observable.fromArray(Object(o.a)(d.map(e=>Object(a.a)(e)))).selectMany(e=>e.catchException(e=>t.Observable.returnValue(e))).toArray().where(e=>!!e.length).select(e=>{throw Object(i.a)(new Error("Suggestion stream error. See inner errors for details."),{innerErrors:e})});return(s(l,u,p,t,{searchText:n})||t.Observable.empty()).concat(g)}const u=e=>e.id},"e5D/":function(e,r,n){"use strict";n.d(r,"i",(function(){return i})),n.d(r,"m",(function(){return c})),n.d(r,"b",(function(){return s})),n.d(r,"w",(function(){return u})),n.d(r,"o",(function(){return l})),n.d(r,"n",(function(){return d})),n.d(r,"a",(function(){return p})),n.d(r,"c",(function(){return g})),n.d(r,"l",(function(){return f})),n.d(r,"t",(function(){return h})),n.d(r,"q",(function(){return m})),n.d(r,"p",(function(){return b})),n.d(r,"r",(function(){return y})),n.d(r,"v",(function(){return v})),n.d(r,"u",(function(){return O})),n.d(r,"s",(function(){return S})),n.d(r,"d",(function(){return j})),n.d(r,"e",(function(){return w})),n.d(r,"g",(function(){return C})),n.d(r,"j",(function(){return P})),n.d(r,"k",(function(){return k})),n.d(r,"f",(function(){return x})),n.d(r,"h",(function(){return E}));var t=function(){for(var e=[],r=0;r<arguments.length;r++)e[r]=arguments[r];for(var n={},t=0,o=e;t<o.length;t++)for(var a=o[t],i=Array.isArray(a)?a:Object.keys(a),c=0,s=i;c<s.length;c++){var u=s[c];n[u]=1}return n},o=t(["onCopy","onCut","onPaste","onCompositionEnd","onCompositionStart","onCompositionUpdate","onFocus","onFocusCapture","onBlur","onBlurCapture","onChange","onInput","onSubmit","onLoad","onError","onKeyDown","onKeyDownCapture","onKeyPress","onKeyUp","onAbort","onCanPlay","onCanPlayThrough","onDurationChange","onEmptied","onEncrypted","onEnded","onLoadedData","onLoadedMetadata","onLoadStart","onPause","onPlay","onPlaying","onProgress","onRateChange","onSeeked","onSeeking","onStalled","onSuspend","onTimeUpdate","onVolumeChange","onWaiting","onClick","onClickCapture","onContextMenu","onDoubleClick","onDrag","onDragEnd","onDragEnter","onDragExit","onDragLeave","onDragOver","onDragStart","onDrop","onMouseDown","onMouseDownCapture","onMouseEnter","onMouseLeave","onMouseMove","onMouseOut","onMouseOver","onMouseUp","onMouseUpCapture","onSelect","onTouchCancel","onTouchEnd","onTouchMove","onTouchStart","onScroll","onWheel","onPointerCancel","onPointerDown","onPointerEnter","onPointerLeave","onPointerMove","onPointerOut","onPointerOver","onPointerUp","onGotPointerCapture","onLostPointerCapture"]),a=t(["accessKey","children","className","contentEditable","dir","draggable","hidden","htmlFor","id","lang","ref","role","style","tabIndex","title","translate","spellCheck","name"]),i=t(a,o),c=t(i,["form"]),s=t(i,["height","loop","muted","preload","src","width"]),u=t(s,["poster"]),l=t(i,["start"]),d=t(i,["value"]),p=t(i,["download","href","hrefLang","media","rel","target","type"]),g=t(i,["autoFocus","disabled","form","formAction","formEncType","formMethod","formNoValidate","formTarget","type","value"]),f=t(g,["accept","alt","autoCapitalize","autoComplete","checked","dirname","form","height","inputMode","list","max","maxLength","min","minLength","multiple","pattern","placeholder","readOnly","required","src","step","size","type","value","width"]),h=t(g,["autoCapitalize","cols","dirname","form","maxLength","minLength","placeholder","readOnly","required","rows","wrap"]),m=t(g,["form","multiple","required"]),b=t(i,["selected","value"]),y=t(i,["cellPadding","cellSpacing"]),v=i,O=t(i,["rowSpan","scope"]),S=t(i,["colSpan","headers","rowSpan","scope"]),j=t(i,["span"]),w=t(i,["span"]),C=t(i,["acceptCharset","action","encType","encType","method","noValidate","target"]),P=t(i,["allow","allowFullScreen","allowPaymentRequest","allowTransparency","csp","height","importance","referrerPolicy","sandbox","src","srcDoc","width"]),k=t(i,["alt","crossOrigin","height","src","srcSet","useMap","width"]),x=i;function E(e,r,n){for(var t=Array.isArray(r),o={},a=0,i=Object.keys(e);a<i.length;a++){var c=i[a];!(!t&&r[c]||t&&r.indexOf(c)>=0||0===c.indexOf("data-")||0===c.indexOf("aria-"))||n&&-1!==(null==n?void 0:n.indexOf(c))||(o[c]=e[c])}return o}},kdYW:function(e,r,n){"use strict";var t,o,a;n.d(r,"c",(function(){return t})),n.d(r,"b",(function(){return o})),n.d(r,"a",(function(){return a})),function(e){e[e.tiny=0]="tiny",e[e.extraExtraSmall=1]="extraExtraSmall",e[e.extraSmall=2]="extraSmall",e[e.small=3]="small",e[e.regular=4]="regular",e[e.large=5]="large",e[e.extraLarge=6]="extraLarge",e[e.size8=17]="size8",e[e.size10=9]="size10",e[e.size16=8]="size16",e[e.size24=10]="size24",e[e.size28=7]="size28",e[e.size32=11]="size32",e[e.size40=12]="size40",e[e.size48=13]="size48",e[e.size56=16]="size56",e[e.size72=14]="size72",e[e.size100=15]="size100",e[e.size120=18]="size120"}(t||(t={})),function(e){e[e.none=0]="none",e[e.offline=1]="offline",e[e.online=2]="online",e[e.away=3]="away",e[e.dnd=4]="dnd",e[e.blocked=5]="blocked",e[e.busy=6]="busy"}(o||(o={})),function(e){e[e.lightBlue=0]="lightBlue",e[e.blue=1]="blue",e[e.darkBlue=2]="darkBlue",e[e.teal=3]="teal",e[e.lightGreen=4]="lightGreen",e[e.green=5]="green",e[e.darkGreen=6]="darkGreen",e[e.lightPink=7]="lightPink",e[e.pink=8]="pink",e[e.magenta=9]="magenta",e[e.purple=10]="purple",e[e.black=11]="black",e[e.orange=12]="orange",e[e.red=13]="red",e[e.darkRed=14]="darkRed",e[e.transparent=15]="transparent",e[e.violet=16]="violet",e[e.lightRed=17]="lightRed",e[e.gold=18]="gold",e[e.burgundy=19]="burgundy",e[e.warmGray=20]="warmGray",e[e.coolGray=21]="coolGray",e[e.gray=22]="gray",e[e.cyan=23]="cyan",e[e.rust=24]="rust"}(a||(a={}))},ksKV:function(e,r,n){"use strict";n.r(r);var t=n("n5ha"),o=n("i8tK"),a=n("GXij"),i=n("GPGK"),c=(Object(t.declareString)("search-dl.quickActions.managePQH.tooltip")),s=Object(i.a)({id:"ManagePQH",entityType:"Text",parameters:[{name:"text",entity:"Text",isRequired:!0,validate:function(e){var r;return!!(null===(r=e.customProps)||void 0===r?void 0:r.isHistory)}}],execute:function(){var e=Object(o.b)();return e?(Object(a.a)(e,"NewTab",window),Promise.resolve()):Promise.reject("No url configured for removePersonalQueryUrl")},additionalProps:{url:function(){return Object(o.b)()},icon:{iconName:"Settings"},tooltip:c}});r.default=s},"oo+0":function(e,r,n){"use strict";n.d(r,"a",(function(){return c}));var t=n("UlRC"),o=n("mkze"),a=n("D7WJ"),i=n("nbvZ");class c{constructor(e,r,n){if(this.config=e,this.groupConfig=r,!r.providers)throw new Error("Search box suggestion type configuration must contain a list of providers.");if(this.providerInstances=r.providers.map(a=>Object(o.a)(a,Object.assign(Object.assign({},n),{config:e,groupConfig:r,rxjs:t}))),!this.providerInstances.every(e=>l(e.getId)&&l(e.createStream)))throw new Error("Suggestion providers must implement methods getId and createStream.")}createStream(e){const r=[],n=[],o=this.providerInstances,c=this.groupConfig.id;for(let u=0;u<o.length;u++)!function(o){try{const u=o.getId();n.push(o.createStream(e).select(e=>Object.assign(Object.assign({},e),{emitTime:Object(a.a)(),groupId:c,providerId:u})).catchException(e=>{var n;return i.a.DEBUG_SB_PROVIDER&&(null===(n=null===console||void 0===console?void 0:console.error)||void 0===n||n.call(console,"In provider:",u,e)),r.push(s(u,e)),t.Observable.empty()}))}catch(e){r.push(s(o.getId(),e))}}(o[u]);return n.push(t.Observable.defer(()=>r.length?t.Observable.throwException(u(c,r)):t.Observable.empty())),Object.freeze(n)}}const s=(e,r)=>{const n=new Error(`Error in suggestion provider "${e}". See inner error for details.`);return n.providerId=e,n.innerError=r,n},u=(e,r)=>{const n=new Error(`Error in suggestion provider group "${e}". See inner error for details.`);return n.providerGroupId=e,n.innerErrors=r,n};function l(e){return"function"==typeof e}},uBHn:function(e,r,n){"use strict";n.d(r,"a",(function(){return i}));var t=n("kdYW"),o=[t.a.lightBlue,t.a.blue,t.a.darkBlue,t.a.teal,t.a.green,t.a.darkGreen,t.a.lightPink,t.a.pink,t.a.magenta,t.a.purple,t.a.orange,t.a.lightRed,t.a.darkRed,t.a.violet,t.a.gold,t.a.burgundy,t.a.warmGray,t.a.cyan,t.a.rust,t.a.coolGray],a=o.length;function i(e){var r=e.primaryText,n=e.text,i=e.initialsColor;return"string"==typeof i?i:function(e){switch(e){case t.a.lightBlue:return"#4F6BED";case t.a.blue:return"#0078D4";case t.a.darkBlue:return"#004E8C";case t.a.teal:return"#038387";case t.a.lightGreen:case t.a.green:return"#498205";case t.a.darkGreen:return"#0B6A0B";case t.a.lightPink:return"#C239B3";case t.a.pink:return"#E3008C";case t.a.magenta:return"#881798";case t.a.purple:return"#5C2E91";case t.a.orange:return"#CA5010";case t.a.red:return"#EE1111";case t.a.lightRed:return"#D13438";case t.a.darkRed:return"#A4262C";case t.a.transparent:return"transparent";case t.a.violet:return"#8764B8";case t.a.gold:return"#986F0B";case t.a.burgundy:return"#750B1C";case t.a.warmGray:return"#7A7574";case t.a.cyan:return"#005B70";case t.a.rust:return"#8E562E";case t.a.coolGray:return"#69797E";case t.a.black:return"#1D1D1D";case t.a.gray:return"#393939"}}(i=void 0!==i?i:function(e){var r=t.a.blue;if(!e)return r;for(var n=0,i=e.length-1;i>=0;i--){var c=e.charCodeAt(i),s=i%8;n^=(c<<s)+(c>>8-s)}return r=o[n%a]}(n||r))}}}]);
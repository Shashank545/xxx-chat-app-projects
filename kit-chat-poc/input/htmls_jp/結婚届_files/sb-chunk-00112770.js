(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["pro-bingtag"],{"+6Wz":function(e,i,t){"use strict";t.d(i,"a",(function(){return T}));var n=t("/tZi"),s=t("+zb2"),o=t("cDcd"),r=t("khAe"),a=t("n8h4"),c=t("dcAx"),l=t("e5D/"),u=t("Cmb2"),d=t("9zDX"),m=t("EYJi"),z=t("27nS"),b=t("fhQA"),f=t("ptyM"),p=t("YyGk"),g=t("kdYW"),h=t("uBHn"),S=t("ZsHU"),O=Object(r.a)({cacheSize:100}),v=Object(a.b)((function(e,i,t,n,s,o){return Object(m.F)(e,!o&&{backgroundColor:Object(h.a)({text:n,initialsColor:i,primaryText:s}),color:t})})),x={size:g.c.size48,presence:g.b.none,imageAlt:""};var _=o.forwardRef((function(e,i){var t=Object(c.a)(x,e),n=function(e){var i=e.onPhotoLoadingStateChange,t=e.imageUrl,n=o.useState(f.c.notLoaded),s=n[0],r=n[1];return o.useEffect((function(){r(f.c.notLoaded)}),[t]),[s,function(e){r(e),null==i||i(e)}]}(t),r=n[0],a=n[1],u=w(a),d=t.className,m=t.coinProps,p=t.showUnknownPersonaCoin,h=t.coinSize,S=t.styles,_=t.imageUrl,y=t.initialsColor,T=t.initialsTextColor,C=t.isOutOfOffice,P=t.onRenderCoin,k=void 0===P?u:P,D=t.onRenderPersonaCoin,N=void 0===D?k:D,A=t.onRenderInitials,E=void 0===A?j:A,I=t.presence,W=t.presenceTitle,R=t.presenceColors,L=t.primaryText,U=t.showInitialsUntilImageLoads,H=t.text,B=t.theme,F=t.size,Q=Object(l.h)(t,l.f),M=Object(l.h)(m||{},l.f),J=h?{width:h,height:h}:void 0,Y=p,Z={coinSize:h,isOutOfOffice:C,presence:I,presenceTitle:W,presenceColors:R,size:F,theme:B},q=O(S,{theme:B,className:m&&m.className?m.className:d,size:F,coinSize:h,showUnknownPersonaCoin:p}),G=Boolean(r!==f.c.loaded&&(U&&_||!_||r===f.c.error||Y));return o.createElement("div",Object(s.__assign)({role:"presentation"},Q,{className:q.coin,ref:i}),F!==g.c.size8&&F!==g.c.size10&&F!==g.c.tiny?o.createElement("div",Object(s.__assign)({role:"presentation"},M,{className:q.imageArea,style:J}),G&&o.createElement("div",{className:v(q.initials,y,T,H,L,p),style:J,"aria-hidden":"true"},E(t,j)),!Y&&N(t,u),o.createElement(z.a,Object(s.__assign)({},Z))):t.presence?o.createElement(z.a,Object(s.__assign)({},Z)):o.createElement(b.a,{iconName:"Contact",className:q.size10WithoutPresenceIcon}),t.children)}));_.displayName="PersonaCoinBase";var w=function(e){return function(i){var t=i.coinSize,n=i.styles,s=i.imageUrl,r=i.imageAlt,a=i.imageShouldFadeIn,c=i.imageShouldStartVisible,l=i.theme,u=i.showUnknownPersonaCoin,d=i.size,m=void 0===d?x.size:d;if(!s)return null;var z=O(n,{theme:l,size:m,showUnknownPersonaCoin:u}),b=t||S.e[m];return o.createElement(p.a,{className:z.image,imageFit:f.b.cover,src:s,width:b,height:b,alt:r,shouldFadeIn:a,shouldStartVisible:c,onLoadingStateChange:e})}},j=function(e){var i=e.imageInitials,t=e.allowPhoneInitials,n=e.showUnknownPersonaCoin,s=e.text,r=e.primaryText,a=e.theme;if(n)return o.createElement(b.a,{iconName:"Help"});var c=Object(u.a)(a);return""!==(i=i||Object(d.a)(s||r||"",c,t))?o.createElement("span",null,i):o.createElement(b.a,{iconName:"Contact"})},y={coin:"ms-Persona-coin",imageArea:"ms-Persona-imageArea",image:"ms-Persona-image",initials:"ms-Persona-initials",size8:"ms-Persona--size8",size10:"ms-Persona--size10",size16:"ms-Persona--size16",size24:"ms-Persona--size24",size28:"ms-Persona--size28",size32:"ms-Persona--size32",size40:"ms-Persona--size40",size48:"ms-Persona--size48",size56:"ms-Persona--size56",size72:"ms-Persona--size72",size100:"ms-Persona--size100",size120:"ms-Persona--size120"},T=Object(n.a)(_,(function(e){var i,t=e.className,n=e.theme,o=e.coinSize,r=n.palette,a=n.fonts,c=Object(S.d)(e.size),l=Object(m.u)(y,n),u=o||e.size&&S.e[e.size]||48;return{coin:[l.coin,a.medium,c.isSize8&&l.size8,c.isSize10&&l.size10,c.isSize16&&l.size16,c.isSize24&&l.size24,c.isSize28&&l.size28,c.isSize32&&l.size32,c.isSize40&&l.size40,c.isSize48&&l.size48,c.isSize56&&l.size56,c.isSize72&&l.size72,c.isSize100&&l.size100,c.isSize120&&l.size120,t],size10WithoutPresenceIcon:{fontSize:a.xSmall.fontSize,position:"absolute",top:"5px",right:"auto",left:0},imageArea:[l.imageArea,{position:"relative",textAlign:"center",flex:"0 0 auto",height:u,width:u},u<=10&&{overflow:"visible",background:"transparent",height:0,width:0}],image:[l.image,{marginRight:"10px",position:"absolute",top:0,left:0,width:"100%",height:"100%",border:0,borderRadius:"50%",perspective:"1px"},u<=10&&{overflow:"visible",background:"transparent",height:0,width:0},u>10&&{height:u,width:u}],initials:[l.initials,{borderRadius:"50%",color:e.showUnknownPersonaCoin?"rgb(168, 0, 0)":r.white,fontSize:a.large.fontSize,fontWeight:m.f.semibold,lineHeight:48===u?46:u,height:u,selectors:(i={},i[m.g]=Object(s.__assign)(Object(s.__assign)({border:"1px solid WindowText"},Object(m.v)()),{color:"WindowText",boxSizing:"border-box",backgroundColor:"Window !important"}),i.i={fontWeight:m.f.semibold},i)},e.showUnknownPersonaCoin&&{backgroundColor:"rgb(234, 234, 234)"},u<32&&{fontSize:a.xSmall.fontSize},u>=32&&u<40&&{fontSize:a.medium.fontSize},u>=40&&u<56&&{fontSize:a.mediumPlus.fontSize},u>=56&&u<72&&{fontSize:a.xLarge.fontSize},u>=72&&u<100&&{fontSize:a.xxLarge.fontSize},u>=100&&{fontSize:a.superLarge.fontSize}]}}),void 0,{scope:"PersonaCoin"})},"00RH":function(e,i,t){"use strict";var n=t("EnCN");i.a=function(e){return(null==e?0:e.length)?Object(n.a)(e,1):[]}},"1txu":function(e,i,t){"use strict";t.r(i);var n=t("+zb2"),s=t("rpSO"),o=t("KmCS"),r=t("F6lS"),a=t("NBkt");let c=class{constructor(e){this.conversation=e}getId(){return"BingConsumerTagSuggestionProvider"}createStream(e){return Object(a.a)({searchText:e,groupName:"PersonalSearchTags",conversationId:this.conversation?this.conversation.conversationId:void 0}).select(i=>({id:i.query||"",url:this.getTagUrl(i),searchText:e,title:i.displayText,displayText:i.displayText||"",fileType:"PersonalSearchTags",iconName:"Tag",onClick:i.sendInstrumentationData}))}getTagUrl(e){var i;return"https://onedrive.live.com/?v=photos&tagFilter="+(0===(null===(i=e.query)||void 0===i?void 0:i.indexOf("#"))?e.query.substr(1):e.query)}};c=Object(n.__decorate)([Object(n.__param)(0,Object(s.a)("conversation"))],c),i.default=c,Object(o.regs)(r.a+".bingConsumerTag",c)},"27nS":function(e,i,t){"use strict";t.d(i,"a",(function(){return O}));var n=t("/tZi"),s=t("cDcd"),o=t("khAe"),r=t("fhQA"),a=t("kdYW"),c=t("ZsHU"),l=t("r3IQ"),u=Object(o.a)({cacheSize:100}),d=s.forwardRef((function(e,i){var t=e.coinSize,n=e.isOutOfOffice,o=e.styles,d=e.presence,z=e.theme,b=e.presenceTitle,f=e.presenceColors,p=s.useRef(null),g=Object(l.a)(i,p),h=Object(c.d)(e.size),S=!(h.isSize8||h.isSize10||h.isSize16||h.isSize24||h.isSize28||h.isSize32)&&(!t||t>32),O=t?t/3<40?t/3+"px":"40px":"",v=t?{fontSize:t?t/6<20?t/6+"px":"20px":"",lineHeight:O}:void 0,x=t?{width:O,height:O}:void 0,_=u(o,{theme:z,presence:d,size:e.size,isOutOfOffice:n,presenceColors:f});return d===a.b.none?null:s.createElement("div",{role:"presentation",className:_.presence,style:x,title:b,ref:g},S&&s.createElement(r.a,{className:_.presenceIcon,iconName:m(e.presence,e.isOutOfOffice),style:v}))}));function m(e,i){if(e){switch(a.b[e]){case"online":return"SkypeCheck";case"away":return i?"SkypeArrow":"SkypeClock";case"dnd":return"SkypeMinus";case"offline":return i?"SkypeArrow":""}return""}}d.displayName="PersonaPresenceBase";var z=t("+zb2"),b=t("EYJi"),f={presence:"ms-Persona-presence",presenceIcon:"ms-Persona-presenceIcon"};function p(e){return{color:e,borderColor:e}}function g(e,i){return{selectors:{":before":{border:"".concat(e," solid ").concat(i)}}}}function h(e){return{height:e,width:e}}function S(e){return{backgroundColor:e}}var O=Object(n.a)(d,(function(e){var i,t,n,s,o,r,a=e.theme,l=e.presenceColors,u=a.semanticColors,d=a.fonts,m=Object(b.u)(f,a),O=Object(c.d)(e.size),v=Object(c.c)(e.presence),x=l&&l.available||"#6BB700",_=l&&l.away||"#FFAA44",w=l&&l.busy||"#C43148",j=l&&l.dnd||"#C50F1F",y=l&&l.offline||"#8A8886",T=l&&l.oof||"#B4009E",C=l&&l.background||u.bodyBackground,P=v.isOffline||e.isOutOfOffice&&(v.isAvailable||v.isBusy||v.isAway||v.isDoNotDisturb),k=O.isSize72||O.isSize100?"2px":"1px";return{presence:[m.presence,Object(z.__assign)(Object(z.__assign)({position:"absolute",height:c.a.size12,width:c.a.size12,borderRadius:"50%",top:"auto",right:"-2px",bottom:"-2px",border:"2px solid ".concat(C),textAlign:"center",boxSizing:"content-box",backgroundClip:"border-box"},Object(b.v)()),{selectors:(i={},i[b.g]={borderColor:"Window",backgroundColor:"WindowText"},i)}),(O.isSize8||O.isSize10)&&{right:"auto",top:"7px",left:0,border:0,selectors:(t={},t[b.g]={top:"9px",border:"1px solid WindowText"},t)},(O.isSize8||O.isSize10||O.isSize24||O.isSize28||O.isSize32)&&h(c.a.size8),(O.isSize40||O.isSize48)&&h(c.a.size12),O.isSize16&&{height:c.a.size6,width:c.a.size6,borderWidth:"1.5px"},O.isSize56&&h(c.a.size16),O.isSize72&&h(c.a.size20),O.isSize100&&h(c.a.size28),O.isSize120&&h(c.a.size32),v.isAvailable&&{backgroundColor:x,selectors:(n={},n[b.g]=S("Highlight"),n)},v.isAway&&S(_),v.isBlocked&&[{selectors:(s={":after":O.isSize40||O.isSize48||O.isSize72||O.isSize100?{content:'""',width:"100%",height:k,backgroundColor:w,transform:"translateY(-50%) rotate(-45deg)",position:"absolute",top:"50%",left:0}:void 0},s[b.g]={selectors:{":after":{width:"calc(100% - 4px)",left:"2px",backgroundColor:"Window"}}},s)}],v.isBusy&&S(w),v.isDoNotDisturb&&S(j),v.isOffline&&S(y),(P||v.isBlocked)&&[{backgroundColor:C,selectors:(o={":before":{content:'""',width:"100%",height:"100%",position:"absolute",top:0,left:0,border:"".concat(k," solid ").concat(w),borderRadius:"50%",boxSizing:"border-box"}},o[b.g]={backgroundColor:"WindowText",selectors:{":before":{width:"calc(100% - 2px)",height:"calc(100% - 2px)",top:"1px",left:"1px",borderColor:"Window"}}},o)}],P&&v.isAvailable&&g(k,x),P&&v.isBusy&&g(k,w),P&&v.isAway&&g(k,T),P&&v.isDoNotDisturb&&g(k,j),P&&v.isOffline&&g(k,y),P&&v.isOffline&&e.isOutOfOffice&&g(k,T)],presenceIcon:[m.presenceIcon,{color:C,fontSize:"6px",lineHeight:c.a.size12,verticalAlign:"top",selectors:(r={},r[b.g]={color:"Window"},r)},O.isSize56&&{fontSize:"8px",lineHeight:c.a.size16},O.isSize72&&{fontSize:d.small.fontSize,lineHeight:c.a.size20},O.isSize100&&{fontSize:d.medium.fontSize,lineHeight:c.a.size28},O.isSize120&&{fontSize:d.medium.fontSize,lineHeight:c.a.size32},v.isAway&&{position:"relative",left:P?void 0:"1px"},P&&v.isAvailable&&p(x),P&&v.isBusy&&p(w),P&&v.isAway&&p(T),P&&v.isDoNotDisturb&&p(j),P&&v.isOffline&&p(y),P&&v.isOffline&&e.isOutOfOffice&&p(T)]}}),void 0,{scope:"PersonaPresence"})},"3WoE":function(e,i,t){"use strict";function n(e){return e>=200&&e<=300||0===e}t.d(i,"a",(function(){return n}))},IRlo:function(e,i,t){"use strict";t.d(i,"a",(function(){return o}));var n=t("zH9T"),s=t("NQus");const o=(e,i)=>Object(n.a)(0,Object(s.a)(""),i,e,[]).dispatch()},O38G:function(e,i,t){"use strict";var n=t("UlRC");n.Observable.prototype.onUnsubscribe=function(e){var i=this;return n.Observable.create((function(t){var n=i.subscribe(t);return function(){e(),n.dispose()}}))},n.Observable.prototype.onSubscribe=function(e){var i=this;return n.Observable.defer((function(){return e(),i}))}},PWby:function(e,i,t){"use strict";var n=t("+zb2"),s=t("UlRC"),o=t("D7WJ"),r=t("TxZW");t("O38G");function a(e){return e&&e.safeToLog?{properties:e.logProperties,error:e.name,message:e.message}:e&&e.name?{error:e.name}:{error:"UnknownError"}}s.Observable.prototype.monitorBasic=function(e,i,t){void 0===t&&(t=a);var n=e.customLogProps?{customLogProps:JSON.stringify(e.customLogProps)}:void 0,s=i?function(e){var t=i(e);return{onNextLogProps:JSON.stringify(t)}}:void 0,o=t?function(e){var i=t(e);return{onErrorLogProps:JSON.stringify(i)}}:void 0;return this.monitor({eventName:"generic_monitor",isHot:e.isHot,nameDetail:e.nameDetail,additionalLogProperties:n,timeSinceOrigin:e.timeSinceOrigin,dispatch:e.dispatch},s,o)},s.Observable.prototype.monitor=function(e,i,t,s){var a,c,l,u,d=null!==(a=e.timeSinceOrigin)&&void 0!==a?a:o.a,m=null!==(c=e.dispatch)&&void 0!==c?c:r.a,z=Math.round(d()),b=-1,f=!1,p={},g=0;e.isHot&&(u=z,m(Object(n.__assign)(Object(n.__assign)({name:e.eventName},e.nameDetail?{nameDetail:e.nameDetail}:{}),{timestamp:z,eventType:"QOSSTART"})));return this.onSubscribe((function(){g++,f||(f=!0,l=Math.round(d()),e.isHot||(u=l,m(Object(n.__assign)(Object(n.__assign)({name:e.eventName},e.nameDetail?{nameDetail:e.nameDetail}:{}),{timestamp:l,eventType:"QOSSTART"}))))})).doAction((function(t){if(b=Math.round(d()),i)try{p=i(t)}catch(i){m(Object(n.__assign)(Object(n.__assign)({eventType:"ERROR",name:e.eventName+"_dataExtraction"},e.nameDetail?{nameDetail:e.nameDetail}:{}),{detail:"An error occurred while extracting data."}))}}),(function(i){var s={};if(t)try{s=t(i)}catch(i){m(Object(n.__assign)(Object(n.__assign)({eventType:"ERROR",name:e.eventName+"_errorDataExtraction"},e.nameDetail?{nameDetail:e.nameDetail}:{}),{detail:"An error occurred while extracting error data."}))}f=!1;var o=Math.round(d());m(Object(n.__assign)(Object(n.__assign)({startTimestamp:u,eventType:"QOSSTOP",name:e.eventName},e.nameDetail?{nameDetail:e.nameDetail}:{}),{result:"FAILURE",totalTime:(b>0?b:o)-u,properties:Object(n.__assign)(Object(n.__assign)(Object(n.__assign)(Object(n.__assign)({},e.additionalLogProperties),p),s),{creationTime:z,subscriptionTime:l,lastItemTime:b,endTime:o}),error:{bucketId:e.eventName+"_exception",detail:e.eventName}})),function(e){throw e}(i)}),(function(){if(f){f=!1;var i=Math.round(d());m(Object(n.__assign)(Object(n.__assign)({startTimestamp:u,totalTime:b-u},e.nameDetail?{nameDetail:e.nameDetail}:{}),{eventType:"QOSSTOP",name:e.eventName,result:s?s():"SUCCESS",properties:Object(n.__assign)(Object(n.__assign)(Object(n.__assign)({},e.additionalLogProperties),p),{creationTime:z,subscriptionTime:l,lastItemTime:b,endTime:i})}))}})).onUnsubscribe((function(){if(g--,f&&!(g>0)){f=!1;var i=Math.round(d());m(Object(n.__assign)(Object(n.__assign)({startTimestamp:u,totalTime:i-u,eventType:"QOSSTOP",name:e.eventName},e.nameDetail?{nameDetail:e.nameDetail}:{}),{result:"CANCELLED",properties:Object(n.__assign)(Object(n.__assign)(Object(n.__assign)({},e.additionalLogProperties),p),{creationTime:z,subscriptionTime:l,lastItemTime:b,endTime:i})})),b=-1}}))}},QlCf:function(e,i,t){"use strict";t.d(i,"d",(function(){return u})),t.d(i,"c",(function(){return d})),t.d(i,"e",(function(){return m})),t.d(i,"f",(function(){return z})),t.d(i,"g",(function(){return g})),t.d(i,"b",(function(){return h}));var n=t("+zb2"),s=t("n8h4"),o=t("m+2y"),r=t("zJH1"),a=Object(s.b)((function(){return Object(o.b)({searchResultFocus:{outline:"1px solid #333333",boxShadow:"0",outlineOffset:"0",textDecoration:"none"},noWrap:{display:"block",textOverflow:"ellipsis",overflow:"hidden",whiteSpace:"nowrap"},offScreen:Object(n.__assign)({clip:"rect(1px, 1px, 1px, 1px)"},{height:"1px",overflow:"hidden",position:"absolute",width:"1px",whiteSpace:"nowrap"}),highlight:{backgroundColor:"#ffee94"}})}),1);i.a=a;var c=function(e){return function(i){var t;return(t={})[e]=i,t}},l=c("@media screen and (-ms-high-contrast: active)"),u=(c("@media (-ms-high-contrast: active), (forced-colors: active)"),c("@media screen and (-ms-high-contrast: white-on-black)")),d=c("@media screen and (-ms-high-contrast: black-on-white)"),m=(Object(n.__assign)(Object(n.__assign)({},{boxSizing:"border-box"}),{boxShadow:"none",margin:0,padding:0}),function(e){var i=e.palette;return Object(n.__assign)(Object(n.__assign)({position:"relative",zIndex:2,display:"inline-block",height:"100% !important",padding:0,margin:0,border:"none",cursor:"pointer",verticalAlign:"top"},p("32px")),{flex:"0 0 ".concat("32px"),fontSize:"16px",selectors:{"&:focus":Object(n.__assign)(Object(n.__assign)({zIndex:3},l({outline:"1px solid ".concat(i.white),backgroundColor:"#1AEBFF",color:i.white})),u({outline:"1px solid ".concat(i.black),backgroundColor:"#37006e",color:i.white}))}})}),z={display:"inline-flex",alignSelf:"center",position:"relative",height:"1em",width:"1em"};var b=Object(r.a)({from:{opacity:0},to:{opacity:1}}),f={animationName:b,animationDuration:"0.167s",animationTimingFunction:"cubic-bezier(.1,.25,.75,.9)",animationFillMode:"both"},p=function(e){return{width:e,minWidth:e,maxWidth:e}},g=f,h={display:"none",opacity:0}},TxZW:function(e,i,t){"use strict";t.d(i,"a",(function(){return s}));var n=t("2CHH");function s(e){Object(n.getDispatcher)().dispatch(e)}},ZsHU:function(e,i,t){"use strict";t.d(i,"b",(function(){return s})),t.d(i,"a",(function(){return o})),t.d(i,"d",(function(){return a})),t.d(i,"e",(function(){return c})),t.d(i,"c",(function(){return l}));var n,s,o,r=t("kdYW");!function(e){e.size8="20px",e.size10="20px",e.size16="16px",e.size24="24px",e.size28="28px",e.size32="32px",e.size40="40px",e.size48="48px",e.size56="56px",e.size72="72px",e.size100="100px",e.size120="120px"}(s||(s={})),function(e){e.size6="6px",e.size8="8px",e.size12="12px",e.size16="16px",e.size20="20px",e.size28="28px",e.size32="32px",e.border="2px"}(o||(o={}));var a=function(e){return{isSize8:e===r.c.size8,isSize10:e===r.c.size10||e===r.c.tiny,isSize16:e===r.c.size16,isSize24:e===r.c.size24||e===r.c.extraExtraSmall,isSize28:e===r.c.size28||e===r.c.extraSmall,isSize32:e===r.c.size32,isSize40:e===r.c.size40||e===r.c.small,isSize48:e===r.c.size48||e===r.c.regular,isSize56:e===r.c.size56,isSize72:e===r.c.size72||e===r.c.large,isSize100:e===r.c.size100||e===r.c.extraLarge,isSize120:e===r.c.size120}},c=((n={})[r.c.tiny]=10,n[r.c.extraExtraSmall]=24,n[r.c.extraSmall]=28,n[r.c.small]=40,n[r.c.regular]=48,n[r.c.large]=72,n[r.c.extraLarge]=100,n[r.c.size8]=8,n[r.c.size10]=10,n[r.c.size16]=16,n[r.c.size24]=24,n[r.c.size28]=28,n[r.c.size32]=32,n[r.c.size40]=40,n[r.c.size48]=48,n[r.c.size56]=56,n[r.c.size72]=72,n[r.c.size100]=100,n[r.c.size120]=120,n),l=function(e){return{isAvailable:e===r.b.online,isAway:e===r.b.away,isBlocked:e===r.b.blocked,isBusy:e===r.b.busy,isDoNotDisturb:e===r.b.dnd,isOffline:e===r.b.offline}}}}]);
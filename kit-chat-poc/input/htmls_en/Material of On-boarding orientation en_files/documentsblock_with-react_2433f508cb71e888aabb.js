(window.webpackJsonpLpc=window.webpackJsonpLpc||[]).push([[204],{1341:function(e,t,n){"use strict";n.r(t),n.d(t,"DocumentsBlockView",(function(){return le}));var r=n(19),i=n.n(r),o=n(45),a=n(46),c=n(67),s=n(72),l=n(44),u=n(1300),p=n(486),d=n(130),f=n(10),m=n(32),b=n(716),h=n(3),v=n(475),g=n(323),O=n(493),j=n(2),y=(Object(j.declareStringWithPlaceholders)("lpc-files-ui.fileListStrings.accessibility.itemPosition"));var x=n(399),C=n(5),w=n.n(C),E=n(1499),k={component:"Files"},L=h.createContext((function(e){return h.createElement(h.Fragment,null,e.children)}));var D=n(1433),S=n(1911),T={name:"File"},N={action:"Open"},I=n(174),F=n(702),A=n(1430),P=n(767),B=n(1409),H=n(1599),R=n(458),M=n(14),K=Object(R.a)((function(e){return{container:{position:"relative",overflow:"hidden",backgroundColor:"transparent",alignItems:"center",justifyContent:"flex-start",padding:"8px 24px",display:"flex",width:"100%",borderBottom:"none",selectors:{"&:hover, .is-focusVisible ":{opacity:1,backgroundColor:e.palette.neutralHover}}},details:[{color:e.palette.neutralPrimary,paddingLeft:"16px",whiteSpace:"nowrap",textOverflow:"ellipsis",overflow:"hidden"},M.j.medium],actionDetails:[{display:"block",color:e.palette.neutralSecondary,whiteSpace:"nowrap",textOverflow:"ellipsis",overflow:"hidden"},M.j.small]}})),U=function(e,t,n){return e?Object(S.a)(I.a.RightToLeft,t,n):n};function W(e){var t=e.title,n=e.onClick,r=e.fileExtension,i=e.subHeading,o=e.useBidiOverride,a=void 0!==o&&o,c=K(Object(P.a)());return h.createElement(F.a,{component:T},h.createElement(A.a,{userAction:N,callback:n},(function(e){return h.createElement(B.a,{onClick:e,className:c.container,title:t},h.createElement(H.a,{fileExtension:r,iconSize:40}),h.createElement("span",{"aria-hidden":!0,className:c.details},h.createElement("span",{title:t},U(a,window,t)),"string"==typeof i?h.createElement("span",{title:i||void 0,className:c.actionDetails},i?U(a,window,i):void 0):i))})))}var V=n(751),z=n(120),_=Object(z.c)((function(){return Object(M.db)({details:{marginLeft:"24px"},ghostLine:[{height:"24px",selectors:{":before":{width:"36ex"}}},M.j.small],ghostLineSecond:[{height:"24px",selectors:{":before":{width:"28ex"}}},M.j.small]})}));function G(){var e=_();return h.createElement("div",{className:e.details},h.createElement(V.a,{className:e.ghostLine}),h.createElement(V.a,{className:e.ghostLineSecond}))}var Y=n(2558);function J(e){var t=e.isConsumer,n=e.isMePersona,r=e.personName,i=function(e){return{noDocumentsDescriptionYouConsumer:Object(m.a)(x.c),noDocumentsDescriptionYou:Object(m.a)(x.b),noDocumentsDescriptionFormat:Object(m.b)(x.a,{p0:e||""})}}(r||"");if(n){var o=t?i.noDocumentsDescriptionYouConsumer:i.noDocumentsDescriptionYou;return h.createElement(Y.a,{message:o})}return r?h.createElement(Y.a,{message:i.noDocumentsDescriptionFormat}):null}var q=n(1410);var Q=function(){return(Q=w.a||function(e){for(var t,n=1,r=arguments.length;n<r;n++)for(var i in t=arguments[n])Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i]);return e}).apply(this,arguments)};function X(e){var t=e.files,n=e.showLoading,r=e.accessibilityListAriaStringFormatter,i=e.personName,o=e.isConsumer,a=e.isMePersona,c=h.useContext(L),s=h.useMemo((function(){return n?[]:function(e,t){var n=t;return e.slice(0,3).map((function(e,t){var r=h.createElement(W,Q({},e,{key:t}));return n?h.createElement(n,{item:e,position:t+1,key:"VerticalContentItemWrapper_".concat(t)},r):r}))}(t,c)}),[n,t]),l=function(e){var t=e.contentItems,n=e.isLoading;return[{properties:{openFileLinkIsEnabled:!0,openFileLinkHasContent:!n&&!!(null==t?void 0:t.length)}},{checkpoints:[{checkpoint:q.a,properties:{openFileLinkHasContent:!n}}]}]}({contentItems:t,isLoading:n}),u=l[0],p=l[1];return Object(D.a)(k,u,p,!1),n?h.createElement(G,null):0===t.length?h.createElement(J,{personName:i,isConsumer:o,isMePersona:a}):h.createElement(E.a,{ariaLabel:r,items:s,itemDisplay:"Vertical"})}function Z(e){var t=e.onTitleClick,n=e.onFooterClick,r=e.onRetryClick,i=e.logName,o=e.files,a=e.showError,c=e.altText,s=e.ariaLabel,l=e.useBidiOverride,u=void 0!==l&&l,p=e.showLoading,d=void 0!==p&&p,f=e.personName,j=e.isConsumer,C=e.isMePersona,w=e.styleOverrides,E=e.hideFooter,k=e.title,L=Object(m.a)(x.e),D=k||L,S=Object(m.a)(x.d),T=0===o.length,N=function(e){var t=Object(g.b)();return function(n,r){var i=e[n-1];return[Object(v.b)(t,y,{itemPosition:"".concat(n),totalCount:"".concat(r)}),O.a[Object(O.d)(i.fileExtension)],i.title].join(". ")}}(o);return h.createElement(b.a,{logName:i,title:D,altText:c,showLoading:d,onHeaderSelected:T?void 0:t,showError:a,footerText:S,onFooterSelected:n,onRetrySelected:r,accessibilityLabel:s,styleOverrides:w,hideFooter:void 0!==E?E:o.length<=3},h.createElement(X,{files:o,showLoading:d,accessibilityListAriaStringFormatter:N,useBidiOverride:u,personName:f,isConsumer:j,isMePersona:C}))}var $=n(1391),ee=n(1479),te=n(132),ne=n(1995),re=n(121),ie=n(205),oe=n(191);function ae(e){var t;return Boolean(e&&e.Title&&Object(O.e)(Object(O.d)(null!==(t=e.FileExtension)&&void 0!==t?t:"")))}var ce=n(424);function se(e){var t=function(){if("undefined"==typeof Reflect||!i.a)return!1;if(i.a.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(i()(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,r=Object(l.a)(e);if(t){var o=Object(l.a)(this).constructor;n=i()(r,arguments,o)}else n=r.apply(this,arguments);return Object(s.a)(this,n)}}var le=function(e){Object(c.a)(n,e);var t=se(n);function n(){var e;return Object(o.a)(this,n),(e=t.apply(this,arguments)).getDocumentsList=Object(z.c)((function(t,n,r,i,o){return(null==t?void 0:t.length)?t.reduce((function(t,a,c){if(ae(a)){var s=e.getDocumentLastActivityString(a,o),l=Object(u.getCompactRelativeTimeString)(Object(p.j)(a.LastActivityTimeStamp)||Object(p.i)(),new Date,Object(ce.b)(i),i.logger),d=o.documentsBlockStrings.timeActionFormat({p0:s,p1:l}),f=n(a.Title||"",a.FileExtension||"",(function(){return r(a,c)}),d);t.push(f)}return t}),[]):[]})),e}return Object(a.a)(n,[{key:"render",value:function(){var e=this,t=this.props.persona,n=this.getDocumentsList(this.props.documents,this.getFileLinks,(function(n,r){return t&&ne.a(n,{indexClicked:r,applicationContext:e.lpcAppContext,logger:e.lpcAppContext.logger})}),this.lpcAppContext,this.strings);return h.createElement(Z,{onTitleClick:function(){return t&&ne.d(t,{applicationContext:e.lpcAppContext,renderingContext:e.props.renderingContext,cardCorrelationId:e.lpcAppContext.cardCorrelationId||"",logger:e.lpcAppContext.logger})},onFooterClick:function(){return t&&ne.b(t,{applicationContext:e.lpcAppContext,renderingContext:e.props.renderingContext,cardCorrelationId:e.lpcAppContext.cardCorrelationId||"",logger:e.lpcAppContext.logger})},onRetryClick:function(){return t&&ne.c(t,e.lpcAppContext,e.props.renderingContext,e.lpcAppContext.logger)},logName:"DocumentsBlock",showError:this.props.hasFailed,ariaLabel:this.strings.contentBlockStrings.documentsBlock,files:n,showLoading:this.props.isLoading,personName:null==t?void 0:t.displayName,isConsumer:Object(oe.c)(this.lpcAppContext.hostAppConfiguration.tenantAadObjectId),isMePersona:this.props.isMePersona})}},{key:"getFileLinks",value:function(e,t,n,r){return{title:e,fileExtension:t,onClick:n,subHeading:r}}},{key:"getDocumentLastActivityString",value:function(e,t){switch(e.LastActivityType){case"Shared":return t.documentsBlockStrings.shared;case"Emailed":return t.documentsBlockStrings.emailed;case"Attached":case"Modified":case"Uploaded":case"Unknown":case void 0:case null:return t.documentsBlockStrings.modified;default:return Object(d.b)(e,t.documentsBlockStrings.modified)}}}]),n}(ie.a);var ue=Object(ee.a)(le,(function(e){var t=e.persona,n=e.hasFailed,r=e.isLoading,i=e.documents;return{componentName:"DocumentsBlock",identifier:t&&t.lpcKey.key||"",logProperties:{hasFailed:n,isLoading:r,personaType:t&&t.kind,documentsCount:i&&i.length.toString()||"0",documentExtensions:i&&i.map((function(e){return e.FileExtension||"unknown"})).toString()}}})),pe=Object(te.c)((function(e,t){var n=t.persona&&Object(re.e)(e,t.persona.lpcKey);return{documents:Object(f.d)(n&&n.files),isLoading:!!n&&n.isLoading,hasFailed:!!n&&n.hasError}}))(Object($.a)(ue));t.default=pe},1459:function(e,t,n){"use strict";n.d(t,"a",(function(){return s}));var r=n(1),i=n(676),o=n(3),a=n(722),c=n(1521);function s(e){var t=Object(a.c)(),n=Object(a.d)(),s=e.identifier,l=e.logProperties,u=n+e.componentName,p=Object(o.useRef)(null),d=Object(o.useRef)(null),f=Object(o.useRef)(null),m=Object(o.useRef)(0);Object(o.useEffect)((function(){return null===p.current&&(p.current=c()),function(){d.current||(d.current=c());var e=p.current&&d.current?d.current-p.current:void 0,n=void 0!==e&&Math.round(e).toString()||"";if(t){var i=Object(r.__assign)({component:u,duration:n,renderCount:m.current.toString()},f.current);t(u,i)}p.current=null,f.current=null,d.current=null,m.current=0}}),[s,u]),Object(o.useEffect)((function(){Object(i.a)(l,f.current)||(m.current++,d.current=c(),f.current=l)}),[u,l])}},1479:function(e,t,n){"use strict";n.d(t,"a",(function(){return a}));var r=n(1),i=n(3),o=n(1459);function a(e,t){return function(n){var a=t(n);return Object(o.a)(a),i.createElement(e,Object(r.__assign)({},n))}}},1492:function(e,t,n){"use strict";n.d(t,"c",(function(){return i})),n.d(t,"b",(function(){return o})),n.d(t,"a",(function(){return a}));var r=n(1668),i=function(e){return"MidgardClipboardUtilsHiddenInput"===e.id},o=function(e,t,n){var r=t||document;return navigator.clipboard?navigator.clipboard.writeText(e).then((function(){return!0})).catch((function(){return c(e,r,n)})):c(e,r,n)},a=function(e,t,n){var i,o=t||document,a=(new r.Converter).makeHtml(e),s=new Blob([a],{type:"text/html"}),l=new Blob([e],{type:"text/plain"}),u=[new ClipboardItem((i={},i["text/plain"]=l,i["text/html"]=s,i))];return navigator.clipboard?navigator.clipboard.write(u).then((function(){return!0})).catch((function(){return c(e,o,n)})):c(e,o,n)},c=function(e,t,n){var r=t.activeElement,i=t.createElement("textarea");i.tabIndex=-1,i.readOnly=!0,i.style.opacity="0",i.style.position="fixed",i.style.right="-9999px",i.style.bottom="-9999px",i.id="MidgardClipboardUtilsHiddenInput";var o=t.createElement("div");o.onselectstart=function(e){return e.stopPropagation()},o.appendChild(i);var a,c=n||t.body;c.appendChild(o),i.value=e,i.focus(),i.select();try{t.execCommand("copy"),a=!0}catch(e){a=!1}return r&&r.focus&&r.focus(),c.removeChild(o),Promise.resolve(a)}},1499:function(e,t,n){"use strict";n.d(t,"b",(function(){return a})),n.d(t,"a",(function(){return c}));var r=n(14),i=n(3),o=Object(r.db)({list:{display:"block",margin:0,padding:0,listStyleType:"none"},listItem:{display:"block"},listItemHorizontal:{display:"inline-block"},verticalDivider:{padding:"0px 4px",flex:"none",flexGrow:0,minWidth:"0px !important",marginTop:"8px",marginBottom:"8px"},listItemHorizontalEnd:{display:"inline-block",flexGrow:1,justifyContent:"flex-end"}}),a="verticalDivider",c=function(e){var t=e.items,n=e.listClassName,r=e.listItemClassName,c=e.itemDisplay,s=e.ariaLabel,l=e.showLastItemOnRight,u=t&&t.filter(Boolean);if(!u||0===u.length)return i.createElement("div",{className:"".concat(o.list," ").concat(n||""),role:"list"});var p=u.map((function(e,t){var n,p;return p="Horizontal"===c?t+1===u.length&&l?o.listItemHorizontalEnd:(null===(n=e.props)||void 0===n?void 0:n.className)===a?o.verticalDivider:o.listItemHorizontal:o.listItem,i.createElement("div",{key:e.key||t,"aria-label":s(t+1,u.length),className:"".concat(p," ").concat(r||"")},e)}));return i.createElement("div",{className:"".concat(o.list," ").concat(n||"")},p)}},1500:function(e,t,n){"use strict";n.d(t,"c",(function(){return a})),n.d(t,"b",(function(){return c})),n.d(t,"d",(function(){return s})),n.d(t,"a",(function(){return l}));var r=n(356),i=n(493),o=n(624);function a(e){return"Mail"===e.ContainerType||"Emailed"===e.LastActivityType||"Shared"===e.LastActivityType}function c(e){var t=i.a[e.Type||""];return Object(i.e)(t)?t:Object(i.d)(function(e){var t=e.Title||"",n=e.FileExtension||"",r=(a(e)?e.WebUrl:e.DownloadUrl)||"";return n.trim()||Object(i.b)(r)||Object(i.b)(t)||""}(e))}function s(e,t){var n=l(e);if(!function(e,t){var n=t.actionProps,r=t.applicationContext,i=t.logger,a=r.hostAppConfiguration.actionCallbacks;if(r.dispatch(Object(o.l)()),null==a?void 0:a.openDocument)try{return a.openDocument(e,n,i),r.dispatch(Object(o.m)()),!0}catch(e){r.dispatch(Object(o.k)({HostAppCallbackNotImplemented:!1}))}return!1}(e,t)){var i={openType:"NewTab",hrefLink:n||""};Object(r.a)(i,t.applicationContext)}}function l(e){return"Mail"===e.ContainerType?e.ContainerWebUrl:e.WebUrl}},1505:function(e,t,n){"use strict";n.d(t,"b",(function(){return i})),n.d(t,"a",(function(){return o}));var r=n(4),i=Object(r.a)("CopyToClipboardSucceededAction",{featureName:"DOMEvent",getLogProperties:function(e){return{targetName:e.targetName}}}),o=Object(r.a)("CopyToClipboardFailedAction",{featureName:"DOMEvent",getLogProperties:function(e){return{targetName:e.targetName}}})},1521:function(e,t,n){(function(t){(function(){var n,r,i,o,a,c;"undefined"!=typeof performance&&null!==performance&&performance.now?e.exports=function(){return performance.now()}:null!=t&&t.hrtime?(e.exports=function(){return(n()-a)/1e6},r=t.hrtime,o=(n=function(){var e;return 1e9*(e=r())[0]+e[1]})(),c=1e9*t.uptime(),a=o-c):Date.now?(e.exports=function(){return Date.now()-i},i=Date.now()):(e.exports=function(){return(new Date).getTime()-i},i=(new Date).getTime())}).call(this)}).call(this,n(1005))},1527:function(e,t,n){"use strict";n.d(t,"a",(function(){return o}));var r=n(1492),i=n(1505);function o(e,t,n){return Object(r.b)(e,n.ownerDocument)?(n.dispatch(Object(i.b)({targetName:t})),!0):(prompt(n.strings.utilityStrings.copyToClipboardHelpText,e),n.dispatch(Object(i.a)({targetName:t})),!1)}},1599:function(e,t,n){"use strict";n.d(t,"a",(function(){return f}));var r=n(1),i=n(2711),o=n(327),a=n(3),c=n(159),s=n(14),l=Object(c.a)((function(e){return Object(s.db)({fileTypeIcon:{flexShrink:0,flexGrow:0,width:"".concat(e,"px"),height:"".concat(e,"px")}})})),u=n(130),p=n(2353);function d(e){if(e)switch(e){case"Folder":case"WebPage":return p.a.folder;default:return Object(u.a)(e)}}var f=function(e){var t=e.fileExtension,n=e.fileType,c=e.iconSize,s=e.className,u=Object(i.a)({extension:t,size:c||32,type:d(n)}),p=l(c||32),f="".concat(p.fileTypeIcon," ").concat(s);return a.createElement(o.a,Object(r.__assign)({role:"presentation"},u,{className:f,"aria-hidden":!0}))}},1911:function(e,t,n){"use strict";n.d(t,"a",(function(){return b}));var r=n(3),i=n(19),o=n.n(i),a=n(45),c=n(46),s=n(67),l=n(72),u=n(44),p=n(174),d=n(371);function f(e){var t=function(){if("undefined"==typeof Reflect||!o.a)return!1;if(o.a.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(o()(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,r=Object(u.a)(e);if(t){var i=Object(u.a)(this).constructor;n=o()(r,arguments,i)}else n=r.apply(this,arguments);return Object(l.a)(this,n)}}var m=function(e){Object(s.a)(n,e);var t=f(n);function n(){return Object(a.a)(this,n),t.apply(this,arguments)}return Object(c.a)(n,[{key:"render",value:function(){var e=Object(d.a)(this.props.ownerWindow),t=this.props.whenDirection===p.a.LeftToRight&&e===p.a.LeftToRight||this.props.whenDirection===p.a.RightToLeft&&e===p.a.RightToLeft?Object(d.b)(e):e;return r.createElement("bdo",{dir:t},this.props.children)}}]),n}(r.Component);function b(e,t,n){return r.createElement(m,{whenDirection:e,ownerWindow:t},n)}},1995:function(e,t,n){"use strict";n.d(t,"d",(function(){return O})),n.d(t,"b",(function(){return j})),n.d(t,"a",(function(){return x})),n.d(t,"f",(function(){return C})),n.d(t,"c",(function(){return w})),n.d(t,"e",(function(){return E})),n.d(t,"i",(function(){return k})),n.d(t,"g",(function(){return L})),n.d(t,"h",(function(){return D}));var r=n(11),i=n.n(r),o=n(9),a=n.n(o),c=n(356),s=n(482),l=n(125),u=n(230),p=n(258),d=n(197),f=n(726),m=n(1500),b=n(488),h=n(1527),v=n(432),g=function(e,t,n,r){return new(n||(n=a.a))((function(i,o){function a(e){try{s(r.next(e))}catch(e){o(e)}}function c(e){try{s(r.throw(e))}catch(e){o(e)}}function s(e){var t;e.done?i(e.value):(t=e.value,t instanceof n?t:new n((function(e){e(t)}))).then(a,c)}s((r=r.apply(e,t||[])).next())}))};function O(e,t){var n=t.applicationContext,r=t.renderingContext,i=y(n),o=Object(p.b)(n),a=Object(d.a)(e)?"GroupFiles":"Files";i&&o?(Object(l.k)(e,t),n.dispatch(Object(u.e)({personaLpcKey:e.lpcKey,isExpandedViewEnabled:!0,section:a,windowId:r.windowId,navigationStartCheckpoint:Object(s.a)(),shouldResetHistory:!1,personaCorrelationId:n.getPersonaCorrelationId(e.lpcKey)})),o.renderExpandedView(n,a)):n.dispatch(Object(u.e)({personaLpcKey:e.lpcKey,isExpandedViewEnabled:!1}))}function j(e,t){var n=t.applicationContext,r=t.renderingContext,i=y(n),o=Object(p.b)(n),a=Object(d.a)(e)?"GroupFiles":"Files";i&&o?(Object(l.k)(e,t),n.dispatch(Object(u.b)({personaLpcKey:e.lpcKey,isExpandedViewEnabled:!0,section:a,windowId:r.windowId,navigationStartCheckpoint:Object(s.a)(),shouldResetHistory:!1,personaCorrelationId:n.getPersonaCorrelationId(e.lpcKey)})),o.renderExpandedView(n,a)):n.dispatch(Object(u.b)({personaLpcKey:e.lpcKey,isExpandedViewEnabled:!1}))}function y(e){var t=Object(p.b)(e);return e.settings.isImmersiveProfileEnabled&&!!t}function x(e,t){var n=t.applicationContext,r=t.indexClicked;n.dispatch(Object(u.a)({document:e,indexClicked:r})),Object(m.d)(e,t)}function C(e,t){t.applicationContext.dispatch(Object(u.g)({document:e})),Object(m.d)(e,t)}function w(e,t,n,r){return g(this,void 0,void 0,i.a.mark((function o(){var a,c;return i.a.wrap((function(i){for(;;)switch(i.prev=i.next){case 0:return t.dispatch(Object(u.c)({personaLpcKey:e.lpcKey})),a=t.reduxStore.getState(),c=f.a.getSelectors(a).getCardTemplateFailed(e.lpcKey),i.next=5,Object(v.a)(e,10,c,t,n,r);case 5:case"end":return i.stop()}}),o)})))}function E(e,t){t.dispatch(Object(u.f)({document:e})),Object(h.a)(e.WebUrl||"","GetPersonaCard",t)}function k(e,t){t.applicationContext.dispatch(Object(u.k)({document:e})),Object(b.b)(e,t)}function L(e,t){t.dispatch(Object(u.h)({document:e}));var n={openType:"NewTab",hrefLink:e.DownloadUrl||""};Object(c.a)(n,t)}function D(e,t){t.dispatch(Object(u.i)({document:e}));var n={openType:"NewTab",hrefLink:e.ContainerWebUrl||""};Object(c.a)(n,t)}},2558:function(e,t,n){"use strict";n.d(t,"a",(function(){return s}));var r=n(767),i=n(3),o=n(458),a=n(14),c=Object(o.a)((function(e){return{noContentStyle:[a.j.small,{padding:"4px 24px",color:e.palette.neutralSecondary,paddingTop:0,paddingBottom:0}]}}));function s(e){var t=e.message,n=c(Object(r.a)()).noContentStyle;return i.createElement("div",{className:n},t)}}}]);
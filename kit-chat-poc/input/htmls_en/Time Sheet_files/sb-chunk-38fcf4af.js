(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["cpl-exit"],{FM7c:function(e,t,r){"use strict";r.r(t);var a=r("6JqZ"),n=r("GtFt"),o=r("f+eq"),s=Object(o.a)({classNames:e=>{var t,r;return e.empty&&!e.active?null===(t=e.styles)||void 0===t?void 0:t.hidden:""+(null===(r=e.styles)||void 0===r?void 0:r.exitButton)},buttonName:n.f}),i=r("zAoU"),c=r("TDQE"),l=r("QlCf"),b=r("n8h4"),h=r("m+2y"),d=r("PS6g"),u=r("EYJi"),p=r("GOdE");var v=Object(b.b)(e=>Object(h.b)({exitButton:["SearchBoxExitButton",c.c]},{exitButton:["SearchBoxExitButton",Object.assign(Object.assign(Object.assign(Object.assign({},Object(p.c)(e)),{backgroundColor:"transparent",margin:"2px",height:"calc(100% - 4px)",color:e.palette.themeDarkAlt,fill:e.palette.themeDarkAlt}),d.a.IconContainer),{"&:hover, &:active":{color:e.palette.themeDark,fill:e.palette.themeDark,backgroundColor:e.palette.neutralLight},[u.g]:{color:"buttontext",fill:"buttontext"}})],icon:Object.assign(Object.assign({},Object(p.a)(e)),{[u.g]:{".SearchBoxExitButton:hover &:first-child":{color:"buttontext",fill:"buttontext"}},selectors:{'html[dir="ltr"] &':{transform:"scale(-1,1)"},".SearchBoxExitButton:hover &:first-child":{fill:e.palette.themeDark}},".SearchBoxExitButton:hover & svg":{stroke:e.palette.themeDarkAlt,strokeWidth:"1px",color:e.palette.themeDarkAlt,fill:e.palette.themeDarkAlt,overflow:"visible"}}),hidden:l.b,visible:l.g}));t.default=Object(a.a)(s,v,i.a)},JjKX:function(e,t,r){"use strict";r.r(t);var a=r("6JqZ"),n=r("zAoU"),o=r("f+eq"),s=r("GtFt"),i=Object(o.a)({classNames:e=>{var t,r;return e.empty&&!e.active?null===(t=e.styles)||void 0===t?void 0:t.hidden:""+(null===(r=e.styles)||void 0===r?void 0:r.exitButton)},buttonName:s.b}),c=r("QlCf"),l=r("n8h4"),b=r("m+2y"),h=r("PS6g"),d=r("GOdE"),u=Object(l.b)(e=>Object(b.b)({exitButton:Object.assign(Object.assign(Object.assign({},Object(d.c)(e)),h.a.IconContainer),{backgroundColor:"transparent",height:"100%",color:e.palette.themeDarkAlt,fill:e.palette.themeDarkAlt,selectors:{":hover":{stroke:e.palette.themeDarkAlt,strokeWidth:"1px",overflow:"visible"}}}),icon:Object.assign(Object.assign({},Object(d.a)(e)),{selectors:{'html[dir="ltr"] &':{transform:"scale(-1,1)"}}}),hidden:c.b}));t.default=Object(a.a)(i,u,n.a)},YUP5:function(e,t,r){"use strict";r.d(t,"a",(function(){return i}));var a=r("UqTr"),n=r("ABrm"),o=r("00RH"),s=r("q9uV");function i(e,t){const r=Object(s.b)(e,t),i=function(e){return Object(a.a)(Object(n.a)(Object(o.a)(e.answers.map(e=>e.providers.map(c)))).map(({entityType:e})=>e))}(e),l=r.reduce((e,t)=>Object.assign(Object.assign({},e),{[t]:!0}),{});for(const e of i)if(!l[e])return i;return r.filter(e=>"BestMatch"!==e)}function c(e){return e.substrateSearchServiceProviderProps}},a4Y0:function(e,t,r){"use strict";r.r(t),r.d(t,"registration",(function(){return v}));var a=r("+zb2"),n=r("rpSO"),o=r("KmCS"),s=r("F6lS"),i=r("POWj"),c=r("ohlq"),l=r("xo1z"),b=r("qGDX"),h=r("q9uV"),d=r("5crk"),u=r("YUP5");let p=class{constructor(e,t,r,a){this.searchBox=e,this.config=t,this.fetcher=r,this.conversation=a,this.conversationId=this.conversation&&this.conversation.conversationId}getId(){return"FactAnswerProvider"}createStream(e){return Object(b.b)(this.fetcher,"People",{searchText:e,entityTypes:Object(u.a)(this.config,e),scenario:Object(l.a)(Object(c.a)()),appOverride:Object(i.a)(),logicalSearchId:this.conversation&&this.conversation.getLogicalSearchId("sb"),conversationId:this.conversationId}).take(1).select(e=>Object.assign(Object.assign({},e),{answerTexts:Object(d.a)(e)})).where(e=>!!e.PeopleIntent&&(e.answerTexts.length>0||"Files"===e.PeopleIntent)).select(t=>({searchText:e,id:t.Id,entityType:"Fact",answerTexts:t.answerTexts,answerConfidence:t.Confidence,personData:t,searchBox:this.searchBox,logProps:t.logProps,originalLogicalSearchId:t.originalLogicalSearchId,originalConversationId:this.conversationId,displayName:t.DisplayName,userPrincipalName:t.UserPrincipalName}))}};p=Object(a.__decorate)([Object(h.a)({entityType:"People"}),Object(a.__param)(0,Object(n.a)("searchBox")),Object(a.__param)(1,Object(n.a)("config")),Object(a.__param)(2,Object(n.a)("fetcher")),Object(a.__param)(3,Object(n.a)("conversation"))],p),t.default=p;const v=Object(o.regs)(`${s.a}.${p.prototype.getId()}`,p)},"f+eq":function(e,t,r){"use strict";var a=r("W4Bk"),n=r("4zpW"),o=r("/Dog"),s=r("cDcd"),i=r("R0Os"),c=r("IRlo"),l=r("JnXn");t.a=({classNames:e,buttonName:t})=>{const r=Object(i.a)(t,"startButton",(function(t,r){var n;return s.createElement("button",{onClick:e=>{var n,o,s,i,l;null===(n=t.onAction)||void 0===n||n.call(t,b(r)),t.onClick&&t.onClick(e),null===(o=t.searchBox)||void 0===o||o.clear(),null===(s=t.searchBox)||void 0===s||s.setFilled(!1),null===(i=t.searchBox)||void 0===i||i.deactivate(),null===(l=t.searchBox)||void 0===l||l.exitSearch(r),(()=>{var e;const r=null===(e=t.instrumenter)||void 0===e?void 0:e.conversationManager,n=null==r?void 0:r.getActiveConversation(),o=(null==n?void 0:n.conversationId)?n:null==r?void 0:r.startConversation(),s=o?o.getLogicalSearchId("sb"):Object(a.a)(),i=o?o.conversationId:Object(a.a)();Object(c.a)(i,s)})()},type:"button",className:e(t),title:t.label,"aria-label":t.label,onMouseDown:l.b,"data-tab":!t.empty&&t.active},s.createElement(o.a,{name:"icon_enter-button",wrapperClassName:null===(n=t.styles)||void 0===n?void 0:n.icon}))})),b=Object(n.a)(r,"onClick");return r}},k9Ik:function(e,t,r){"use strict";r.r(t),r.d(t,"registration",(function(){return v}));var a=r("+zb2"),n=r("rpSO"),o=r("KmCS"),s=r("F6lS"),i=r("POWj"),c=r("ohlq"),l=r("xo1z"),b=r("qGDX"),h=r("q9uV"),d=r("5crk"),u=r("YUP5");let p=class{constructor(e,t,r,a,n){this.instrumenter=t,this.searchBox=e,this.config=r,this.fetcher=a,this.conversation=n}getId(){return"PersonAnswerProvider"}createStream(e){return Object(b.b)(this.fetcher,"People",{searchText:e,entityTypes:Object(u.a)(this.config,e),scenario:Object(l.a)(Object(c.a)()),appOverride:Object(i.a)(),logicalSearchId:this.conversation&&this.conversation.getLogicalSearchId("sb"),conversationId:this.conversation&&this.conversation.conversationId}).take(1).defaultIfEmpty([]).select(e=>{var t,r,a;return this.isPersonAnswerScoped()&&Array.isArray(e)&&!this.isPersonPilled()&&(null===(t=this.instrumenter)||void 0===t||t.setProps({hostAppInfo:Object.assign(Object.assign({},null===(a=null===(r=this.instrumenter)||void 0===r?void 0:r.props)||void 0===a?void 0:a.hostAppInfo),{scopeId:""})})),Object.assign(Object.assign({},e),{answerTexts:Object(d.a)(e)})}).select(t=>({searchText:e,id:t.Id,entityType:"Person",answerTexts:t.answerTexts,answerConfidence:t.Confidence,jobTitle:t.JobTitle,personData:t,searchBox:this.searchBox,config:this.config,displayName:t.DisplayName,userPrincipalName:t.UserPrincipalName}))}isPersonAnswerScoped(){var e,t;return"personAnswer"===(null===(t=null===(e=this.instrumenter)||void 0===e?void 0:e.props)||void 0===t?void 0:t.hostAppInfo.scopeId)}isPersonPilled(){var e,t,r;return!!(null===(r=null===(t=null===(e=this.instrumenter)||void 0===e?void 0:e.props)||void 0===t?void 0:t.scenarioContext)||void 0===r?void 0:r.personScope)}};p=Object(a.__decorate)([Object(h.a)({entityType:"People"}),Object(a.__param)(0,Object(n.a)("searchBox")),Object(a.__param)(1,Object(n.a)("instrumenter")),Object(a.__param)(2,Object(n.a)("config")),Object(a.__param)(3,Object(n.a)("fetcher")),Object(a.__param)(4,Object(n.a)("conversation"))],p),t.default=p;const v=Object(o.regs)(`${s.a}.${p.prototype.getId()}`,p)},zAoU:function(e,t,r){"use strict";r.d(t,"a",(function(){return n}));var a=r("n5ha");const n=(Object(a.declareString)("search-box-container-plugins.searchux.strings.SearchBoxExitButton.label"))}}]);
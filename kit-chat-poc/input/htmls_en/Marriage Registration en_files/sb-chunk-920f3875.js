(window.msfast_searchux_sb_jsonp=window.msfast_searchux_sb_jsonp||[]).push([["cpl-meetingpill"],{"6AAz":function(e,t){var a,n,r=e.exports={};function i(){throw new Error("setTimeout has not been defined")}function o(){throw new Error("clearTimeout has not been defined")}function s(e){if(a===setTimeout)return setTimeout(e,0);if((a===i||!a)&&setTimeout)return a=setTimeout,setTimeout(e,0);try{return a(e,0)}catch(t){try{return a.call(null,e,0)}catch(t){return a.call(this,e,0)}}}!function(){try{a="function"==typeof setTimeout?setTimeout:i}catch(e){a=i}try{n="function"==typeof clearTimeout?clearTimeout:o}catch(e){n=o}}();var c,l=[],m=!1,d=-1;function u(){m&&c&&(m=!1,c.length?l=c.concat(l):d=-1,l.length&&h())}function h(){if(!m){var e=s(u);m=!0;for(var t=l.length;t;){for(c=l,l=[];++d<t;)c&&c[d].run();d=-1,t=l.length}c=null,m=!1,function(e){if(n===clearTimeout)return clearTimeout(e);if((n===o||!n)&&clearTimeout)return n=clearTimeout,clearTimeout(e);try{n(e)}catch(t){try{return n.call(null,e)}catch(t){return n.call(this,e)}}}(e)}}function p(e,t){this.fun=e,this.array=t}function g(){}r.nextTick=function(e){var t=new Array(arguments.length-1);if(arguments.length>1)for(var a=1;a<arguments.length;a++)t[a-1]=arguments[a];l.push(new p(e,t)),1!==l.length||m||s(h)},p.prototype.run=function(){this.fun.apply(null,this.array)},r.title="browser",r.browser=!0,r.env={},r.argv=[],r.version="",r.versions={},r.on=g,r.addListener=g,r.once=g,r.off=g,r.removeListener=g,r.removeAllListeners=g,r.emit=g,r.prependListener=g,r.prependOnceListener=g,r.listeners=function(e){return[]},r.binding=function(e){throw new Error("process.binding is not supported")},r.cwd=function(){return"/"},r.chdir=function(e){throw new Error("process.chdir is not supported")},r.umask=function(){return 0}},Ytuu:function(e,t,a){"use strict";a.r(t),a.d(t,"registration",(function(){return p}));var n=a("+zb2"),r=a("rpSO"),i=a("KmCS"),o=a("F6lS"),s=a("POWj"),c=a("ohlq"),l=a("xo1z"),m=a("qGDX"),d=a("q9uV"),u=a("5crk");let h=class{constructor(e,t,a,n){this.searchBox=e,this.config=t,this.fetcher=a,this.conversation=n,this.conversationId=this.conversation&&this.conversation.conversationId}getId(){return"AnswerSuggestionProvider"}createStream(e){return Object(m.b)(this.fetcher,"People",{searchText:e,entityTypes:Object(d.b)(this.config,e),scenario:Object(l.a)(Object(c.a)()),appOverride:Object(s.a)(),logicalSearchId:this.conversation&&this.conversation.getLogicalSearchId("sb"),conversationId:this.conversationId}).take(1).select(e=>Object.assign(Object.assign({},e),{answerTexts:Object(u.a)(e)})).where(e=>!!e.PeopleIntent&&e.answerTexts.length>0).select(t=>({searchText:e,searchBox:this.searchBox,id:t.Id,answerTexts:t.answerTexts,jobTitle:t.JobTitle,displayName:t.DisplayName,userPrincipalName:t.UserPrincipalName,peopleIntent:t.PeopleIntent,logProps:t.logProps,originalLogicalSearchId:t.originalLogicalSearchId,originalConversationId:this.conversationId,entityType:"People"}))}};h=Object(n.__decorate)([Object(d.a)({entityType:"People"}),Object(n.__param)(0,Object(r.a)("searchBox")),Object(n.__param)(1,Object(r.a)("config")),Object(n.__param)(2,Object(r.a)("fetcher")),Object(n.__param)(3,Object(r.a)("conversation"))],h),t.default=h;const p=Object(i.regs)(`${o.a}.${h.prototype.getId()}`,h)},YyGk:function(e,t,a){"use strict";a.d(t,"a",(function(){return b}));var n=a("/tZi"),r=a("+zb2"),i=a("cDcd"),o=a("khAe"),s=a("e5D/"),c=a("ptyM"),l=a("M3Ou"),m=a("r3IQ"),d=Object(o.a)(),u=/\.svg$/i;var h=i.forwardRef((function(e,t){var a=i.useRef(),n=i.useRef(),o=function(e,t){var a=e.onLoadingStateChange,n=e.onLoad,r=e.onError,o=e.src,s=i.useState(c.c.notLoaded),m=s[0],d=s[1];Object(l.a)((function(){d(c.c.notLoaded)}),[o]),i.useEffect((function(){m===c.c.notLoaded&&(!!t.current&&(o&&t.current.naturalWidth>0&&t.current.naturalHeight>0||t.current.complete&&u.test(o))&&d(c.c.loaded))})),i.useEffect((function(){null==a||a(m)}),[m]);var h=i.useCallback((function(e){null==n||n(e),o&&d(c.c.loaded)}),[o,n]),p=i.useCallback((function(e){null==r||r(e),d(c.c.error)}),[r]);return[m,h,p]}(e,n),h=o[0],p=o[1],g=o[2],f=Object(s.h)(e,s.k,["width","height"]),b=e.src,v=e.alt,I=e.width,C=e.height,j=e.shouldFadeIn,O=void 0===j||j,y=e.shouldStartVisible,w=e.className,x=e.imageFit,P=e.role,L=e.maximizeFrame,_=e.styles,S=e.theme,T=e.loading,N=function(e,t,a,n){var r=i.useRef(t),o=i.useRef();(void 0===o||r.current===c.c.notLoaded&&t===c.c.loaded)&&(o.current=function(e,t,a,n){var r=e.imageFit,i=e.width,o=e.height;if(void 0!==e.coverStyle)return e.coverStyle;if(t===c.c.loaded&&(r===c.b.cover||r===c.b.contain||r===c.b.centerContain||r===c.b.centerCover)&&a.current&&n.current){var s=void 0;if(s="number"==typeof i&&"number"==typeof o&&r!==c.b.centerContain&&r!==c.b.centerCover?i/o:n.current.clientWidth/n.current.clientHeight,a.current.naturalWidth/a.current.naturalHeight>s)return c.a.landscape}return c.a.portrait}(e,t,a,n));return r.current=t,o.current}(e,h,n,a),E=d(_,{theme:S,className:w,width:I,height:C,maximizeFrame:L,shouldFadeIn:O,shouldStartVisible:y,isLoaded:h===c.c.loaded||h===c.c.notLoaded&&e.shouldStartVisible,isLandscape:N===c.a.landscape,isCenter:x===c.b.center,isCenterContain:x===c.b.centerContain,isCenterCover:x===c.b.centerCover,isContain:x===c.b.contain,isCover:x===c.b.cover,isNone:x===c.b.none,isError:h===c.c.error,isNotImageFit:void 0===x});return i.createElement("div",{className:E.root,style:{width:I,height:C},ref:a},i.createElement("img",Object(r.__assign)({},f,{onLoad:p,onError:g,key:"fabricImage"+e.src||"",className:E.image,ref:Object(m.a)(n,t),src:b,alt:v,role:P,loading:T})))}));h.displayName="ImageBase";var p=a("EYJi"),g=a("kumM"),f={root:"ms-Image",rootMaximizeFrame:"ms-Image--maximizeFrame",image:"ms-Image-image",imageCenter:"ms-Image-image--center",imageContain:"ms-Image-image--contain",imageCover:"ms-Image-image--cover",imageCenterContain:"ms-Image-image--centerContain",imageCenterCover:"ms-Image-image--centerCover",imageNone:"ms-Image-image--none",imageLandscape:"ms-Image-image--landscape",imagePortrait:"ms-Image-image--portrait"},b=Object(n.a)(h,(function(e){var t=e.className,a=e.width,n=e.height,r=e.maximizeFrame,i=e.isLoaded,o=e.shouldFadeIn,s=e.shouldStartVisible,c=e.isLandscape,l=e.isCenter,m=e.isContain,d=e.isCover,u=e.isCenterContain,h=e.isCenterCover,b=e.isNone,v=e.isError,I=e.isNotImageFit,C=e.theme,j=Object(p.u)(f,C),O={position:"absolute",left:"50% /* @noflip */",top:"50%",transform:"translate(-50%,-50%)"},y=Object(g.a)(),w=void 0!==y&&void 0===y.navigator.msMaxTouchPoints,x=m&&c||d&&!c?{width:"100%",height:"auto"}:{width:"auto",height:"100%"};return{root:[j.root,C.fonts.medium,{overflow:"hidden"},r&&[j.rootMaximizeFrame,{height:"100%",width:"100%"}],i&&o&&!s&&p.a.fadeIn400,(l||m||d||u||h)&&{position:"relative"},t],image:[j.image,{display:"block",opacity:0},i&&["is-loaded",{opacity:1}],l&&[j.imageCenter,O],m&&[j.imageContain,w&&{width:"100%",height:"100%",objectFit:"contain"},!w&&x,!w&&O],d&&[j.imageCover,w&&{width:"100%",height:"100%",objectFit:"cover"},!w&&x,!w&&O],u&&[j.imageCenterContain,c&&{maxWidth:"100%"},!c&&{maxHeight:"100%"},O],h&&[j.imageCenterCover,c&&{maxHeight:"100%"},!c&&{maxWidth:"100%"},O],b&&[j.imageNone,{width:"auto",height:"auto"}],I&&[!!a&&!n&&{height:"auto",width:"100%"},!a&&!!n&&{height:"100%",width:"auto"},!!a&&!!n&&{height:"100%",width:"100%"}],c&&j.imageLandscape,!c&&j.imagePortrait,!i&&"is-notLoaded",o&&"is-fadeIn",v&&"is-error"]}}),void 0,{scope:"Image"},!0);b.displayName="Image"},aq8x:function(e,t,a){"use strict";a.r(t);var n=a("wAGM"),r=a("GtFt"),i=a("3twC"),o=a("D59/"),s=a("oQg0"),c=a("cDcd");const l=Object(o.a)("mp-comp",()=>Promise.all([a.e("strings"),a.e("pp-comp"),a.e("commands"),a.e(2),a.e("sug-news"),a.e(3),a.e("cmd-copyfileurl"),a.e("sug-copilot"),a.e(4),a.e("cpl-search"),a.e("cpl-meetingpill"),a.e("cpl-host"),a.e("cmd-deletepqh"),a.e("pro-bingtag"),a.e(5),a.e("vendors~cpl-teachingmoments~debug~fb-comp~mp-comp~pp-comp~sc-comp~sug-pplkey~tp-comp"),a.e("cmd-openchat"),a.e("pro-3s-pplkw"),a.e("cmd-copyemail"),a.e("mp-comp")]).then(a.bind(null,"tYNB")),e=>e.default),m=Object(s.a)(l);class d extends c.Component{constructor(e){var t,a,n;super(e),this.state={meetingPillData:(null===(n=null===(a=null===(t=this.props.instrumenter)||void 0===t?void 0:t.props)||void 0===a?void 0:a.scenarioContext)||void 0===n?void 0:n.meetingScope)||{}}}componentDidMount(){var e;const t=null===(e=this.props.instrumenter)||void 0===e?void 0:e.register(e=>{var t;e.hostAppInfo&&("meeting"===e.hostAppInfo.scopeId&&(null===(t=e.scenarioContext)||void 0===t?void 0:t.meetingScope)?this.setState({meetingPillData:e.scenarioContext.meetingScope}):this.setState({meetingPillData:{}}))});this.setState({instrumenterId:t})}render(){return this.state.meetingPillData&&this.state.meetingPillData.meetingId?c.createElement(m,Object.assign({},this.props,{meetingPillData:this.state.meetingPillData||{}})):null}}var u=Object(i.a)("startButton","SearchBoxMeetingPill")(e=>c.createElement(d,Object.assign({},e)));t.default=Object(n.a)(r.m)(u)},d9Ub:function(e,t,a){"use strict";a.r(t),a.d(t,"registration",(function(){return m}));var n=a("+zb2"),r=a("UlRC"),i=a("rpSO"),o=a("KmCS"),s=a("F6lS"),c=a("q9uV");let l=class{constructor(e,t,a){this.searchBox=e,this.conversationId=t&&t.conversationId,this.context=a}getId(){return"AnswerPersonDataProvider"}createStream(e){var t,a;const n=null===(a=null===(t=this.context)||void 0===t?void 0:t.commanding.currentCommand.values[0])||void 0===a?void 0:a.customProps;return r.Observable.fromPromise(Promise.resolve(n)).select(t=>({searchText:e,searchBox:this.searchBox,id:t.id,answerTexts:n.answerTexts,jobTitle:n.jobTitle,displayName:n.displayName,userPrincipalName:null==n?void 0:n.userPrincipalName,peopleIntent:null==n?void 0:n.peopleIntent,logProps:t.logProps,originalLogicalSearchId:t.originalLogicalSearchId,originalConversationId:this.conversationId}))}};l=Object(n.__decorate)([Object(c.a)({entityType:"People"}),Object(n.__param)(0,Object(i.a)("searchBox")),Object(n.__param)(1,Object(i.a)("conversation")),Object(n.__param)(2,Object(i.a)("context"))],l),t.default=l;const m=Object(o.regs)(`${s.a}.${l.prototype.getId()}`,l)},fhQA:function(e,t,a){"use strict";a.d(t,"a",(function(){return g}));var n,r=a("/tZi"),i=a("+zb2"),o=a("cDcd");!function(e){e[e.default=0]="default",e[e.image=1]="image",e[e.Default=1e5]="Default",e[e.Image=100001]="Image"}(n||(n={}));var s=a("YyGk"),c=a("ptyM"),l=a("khAe"),m=a("e5D/"),d=a("cOxX"),u=Object(l.a)({cacheSize:100}),h=function(e){function t(t){var a=e.call(this,t)||this;return a._onImageLoadingStateChange=function(e){a.props.imageProps&&a.props.imageProps.onLoadingStateChange&&a.props.imageProps.onLoadingStateChange(e),e===c.c.error&&a.setState({imageLoadError:!0})},a.state={imageLoadError:!1},a}return Object(i.__extends)(t,e),t.prototype.render=function(){var e=this.props,t=e.children,a=e.className,r=e.styles,c=e.iconName,l=e.imageErrorAs,h=e.theme,p="string"==typeof c&&0===c.length,g=!!this.props.imageProps||this.props.iconType===n.image||this.props.iconType===n.Image,f=Object(d.b)(c)||{},b=f.iconClassName,v=f.children,I=f.mergeImageProps,C=u(r,{theme:h,className:a,iconClassName:b,isImage:g,isPlaceholder:p}),j=g?"span":"i",O=Object(m.h)(this.props,m.i,["aria-label"]),y=this.state.imageLoadError,w=Object(i.__assign)(Object(i.__assign)({},this.props.imageProps),{onLoadingStateChange:this._onImageLoadingStateChange}),x=y&&l||s.a,P=this.props["aria-label"]||this.props.ariaLabel,L=w.alt||P||this.props.title,_=!!(L||this.props["aria-labelledby"]||w["aria-label"]||w["aria-labelledby"])?{role:g||I?void 0:"img","aria-label":g||I?void 0:L}:{"aria-hidden":!0},S=v;return I&&v&&"object"==typeof v&&L&&(S=o.cloneElement(v,{alt:L})),o.createElement(j,Object(i.__assign)({"data-icon-name":c},_,O,I?{title:void 0,"aria-label":void 0}:{},{className:C.root}),g?o.createElement(x,Object(i.__assign)({},w)):t||S)},t}(o.Component),p=a("njod"),g=Object(r.a)(h,p.c,void 0,{scope:"Icon"},!0);g.displayName="Icon"},ptyM:function(e,t,a){"use strict";var n,r,i;a.d(t,"b",(function(){return n})),a.d(t,"a",(function(){return r})),a.d(t,"c",(function(){return i})),function(e){e[e.center=0]="center",e[e.contain=1]="contain",e[e.cover=2]="cover",e[e.none=3]="none",e[e.centerCover=4]="centerCover",e[e.centerContain=5]="centerContain"}(n||(n={})),function(e){e[e.landscape=0]="landscape",e[e.portrait=1]="portrait"}(r||(r={})),function(e){e[e.notLoaded=0]="notLoaded",e[e.loaded=1]="loaded",e[e.error=2]="error",e[e.errorLoaded=3]="errorLoaded"}(i||(i={}))}}]);
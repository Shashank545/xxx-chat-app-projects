define("b19b3b9e-8d13-4fec-a93c-401a091c0707_0.1.0",["tslib","@microsoft/sp-http-base","@ms/sp-telemetry","@ms/office-ui-fabric-react-bundle","@ms/sp-webpart-shared","@microsoft/sp-core-library","@microsoft/sp-page-context","@microsoft/sp-webpart-base","react","react-dom","@microsoft/load-themed-styles","@microsoft/sp-diagnostics","@microsoft/sp-http","@ms/odsp-utilities-bundle","@ms/sp-a11y"],function(n,a,i,r,o,s,c,d,l,u,f,p,m,_,h){return function(e){var t={};function n(a){if(t[a])return t[a].exports;var i=t[a]={i:a,l:!1,exports:{}};return e[a].call(i.exports,i,i.exports,n),i.l=!0,i.exports}return n.m=e,n.c=t,n.d=function(e,t,a){n.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:a})},n.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},n.t=function(e,t){if(1&t&&(e=n(e)),8&t)return e;if(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var a=Object.create(null);if(n.r(a),Object.defineProperty(a,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var i in e)n.d(a,i,function(t){return e[t]}.bind(null,i));return a},n.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return n.d(t,"a",t),t},n.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},n.p="",n(n.s="Fqzb")}({"17wl":function(e,t){e.exports=n},"2SXB":function(e,t){e.exports=a},"2q6Q":function(e,t){e.exports=i},"5pbq":function(e,t){e.exports=r},Fqzb:function(e,t,n){"use strict";n.r(t);var a=n("17wl"),i=n("UWqr"),r=n("2q6Q"),o=n("ut3N"),s=n("br4S"),c=n("ytfe"),d=n("y88i"),l=n("U4ag"),u=n("5pbq"),f=n("cDcd"),p=n("faye"),m=n("vlQI"),_=n("2SXB"),h=n("X+PM"),b=n("mGD9"),g=function(){function e(){}return e.getIncorrectDomainErrorMessage=function(e){if(e&&!/^https:\/\/forms.(office-int.com\/.*|officeppe.com\/.*|office.com\/.*)/i.test(e))return b.F},e.getEmptyUrlErrorMessage=function(e){return e?"":b.T},e.isValidFormUrl=function(e){if(e){var t=new d.Uri(e).getHost();if(t){var n=t.toLowerCase();return"forms.office.com"===n||"forms.officeppe.com"===n}}return!1},e.getInvalidValidFormIdErrorMessage=function(e){return e&&/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/g.test(e.replace(/\-/g,"+").replace(/\_/g,"/"))?void 0:b.M},e.isValidTitle=function(e){return!!e&&e.length<=90},e.getCoAuthTokenFromFormUrl=function(t){var n,a=new d.Uri(t,{queryCaseInsensitive:!0});if("designpage.aspx"===a.getLastPathSegment().toLowerCase()){var i=a.getFragment(),r=e._getHashes(i),o=a.getQueryParameter("fragment"),s=e._getHashes(o);n=r.has("token")?r.get("token"):s.has("token")?s.get("token"):void 0}return n},e.getDisplayTypeFromFormUrl=function(e){return e&&"analysispage.aspx"===new d.Uri(e).getLastPathSegment().toLowerCase()?2:1},e.pastedURLType=function(e){return/.*DesignPage.*/g.test(e)?"designPage":/.*ResponsePage.*/g.test(e)?"responsePage":/.*AnalysisPage.*/g.test(e)?"analysisPage":"others"},e.getFormIdFromUrl=function(t){var n,a=new d.Uri(t,{queryCaseInsensitive:!0}),i=a.getLastPathSegment().toLowerCase();if("responsepage.aspx"===i||"analysispage.aspx"===i)n=a.getQueryParameter("id");else if("designpage.aspx"===i){var r=a.getFragment(),o=e._getHashes(r),s=a.getQueryParameter("fragment"),c=e._getHashes(s);n=o.has("formid")?o.get("formid"):c.has("formid")?c.get("formid"):void 0}return n||""},e.decodeFormId=function(t){var n=new r._QosMonitor("FormsUtil.decodeFormId");try{t=t.replace(/_/g,"/").replace(/-/g,"+");var a=e._base64StringToBytes(t),i=a.slice(0,e._orgIdSize),o=e._bytesToGuidString(i),s=a.slice(e._orgIdSize,e._orgIdSize+e._ownerIdSize),c=e._bytesToGuidString(s),d=a.slice(e._orgIdSize+e._ownerIdSize),l=e._stringTrimEndChars(e._bytesToString(d),e._padding),u={},f=l.split(e._separator);if(f.length>1){l=f[0];for(var p=1;p<f.length;p++){var m=f[p].split("=");m&&2===m.length&&(u[m[0]]=m[1])}}var _=u&&u[e._ownerTypeKey]===e._ownerTypeGroup;return n.writeSuccess(),{orgId:o,ownerId:c,tableId:l,isGroupOwner:_}}catch(e){return n.writeUnexpectedFailure("FailedToDecodeFormId",e),{orgId:void 0,ownerId:void 0,tableId:void 0,isGroupOwner:!1}}},e.getErrorMessageFromError=function(e){switch(e.message){case"750":case"751":case"752":case"753":case"754":return b.E;case"701":case"707":case"708":case"709":case"5555":return b.L;case"6212":return b.A;default:return b.P}},e.getFormThemeColor=function(e){var t="6C6C6C";if(!e)return t;var n=window.getComputedStyle(e);if(!n||!n.color)return t;var a=n.color.match(/\d+/g);if(3!==(null==a?void 0:a.length))return t;var i=a.map(function(e){return parseInt(e,10)});return this._bytesToHexString(i)},e.isValidShortUrlFormat=function(t){if(e.getEmptyUrlErrorMessage(t))return!1;if(e.getIncorrectDomainErrorMessage(t))return!1;var n=new d.Uri(t,{queryCaseInsensitive:!0}).getPath();return/^\/[rge]\/[\w]+$/gi.test(n)},e._base64StringToBytes=function(t){return e._stringToAsciiBytes(window.atob(t.replace(/ /g,"")))},e._bytesToGuidString=function(t){if(16===t.length){var n=[].concat(t.slice(0,4).reverse(),t.slice(4,6).reverse(),t.slice(6,8).reverse(),t.slice(8)),a=e._bytesToHexString(n);return"".concat(a.slice(0,8),"-").concat(a.slice(8,12),"-").concat(a.slice(12,16),"-").concat(a.slice(16,20),"-").concat(a.slice(20))}},e._bytesToHexString=function(e){for(var t="",n="",a=0;a<e.length;a++)t+=n=1===(n=e[a].toString(16)).length?"0"+n:n;return t},e._bytesToString=function(e){for(var t="",n=0;n<e.length;++n)t+=String.fromCharCode(e[n]);return t},e._getHashes=function(e){var t=new Map;return e&&e.trim().split("&").forEach(function(e){var n=e.split("="),a=n[0].trim().toLowerCase();if(a){var i=decodeURIComponent(n[1]||"").trim();t.set(a,i)}}),t},e._stringToAsciiBytes=function(e){return e.split("").map(function(e){return e.charCodeAt(0)})},e._stringTrimEndChars=function(e,t){if(!e)return e;for(var n=e.length;--n>=0&&!(t.indexOf(e.charAt(n))<0););return e.substr(0,n+1)},e._ownerTypeKey="t",e._ownerTypeGroup="g",e._orgIdSize=16,e._ownerIdSize=16,e._padding=".",e._separator="$%@#",e}();function v(){return i._SPFlight.isEnabled(1725)}function y(){return i._SPKillSwitch.isActivated("1524f79e-bd10-4f74-8ea1-c59c3093f235")}var S=function(){function e(e){var t=this;e.whenFinished(function(){t._httpClient=new m.HttpClient(e),t._context=e.consume(h.PageContext.serviceKey),v()&&!y()?t._aadTokenProvider=new m.AadTokenProvider(_._AadTokenProviders.preAuthorizedConfiguration):t._tokenProvider=e.consume(m.DEPRECATED_DO_NOT_USE_OAuthTokenProvider.serviceKey)})}return e.prototype.createForm=function(e,t){var n=new r._QosMonitor("FormsDataSource.CreateForm"),a=t?"groups":"users",i="".concat(this._baseEndpoint,"/formapi/api/").concat(e.ownerTenantId,"/").concat(a,"/").concat(e.ownerId,"/forms");return this._sendRequest("POST",i,e).then(function(e){return n.writeSuccess(),e}).catch(function(e){return n.writeUnexpectedFailure("CreateFormFailure",e),Promise.reject(e)})},e.prototype.getAccessToken=function(){return this._getOAuthToken().then(function(e){return e})},e.prototype.getForms=function(e,t,n){var a=new r._QosMonitor("FormsDataSource.GetForms"),i=n?"groups":"users",o="".concat(this._baseEndpoint,"/formapi/api/").concat(e,"/").concat(i,"/").concat(t,"/forms");return this._sendRequest("GET",o).then(function(e){return a.writeSuccess(),e.value}).catch(function(e){return a.writeUnexpectedFailure("GetFormFailure",e),Promise.reject(e)})},e.prototype.getOrCreatePermissionToken=function(e,t){var n=new r._QosMonitor("FormsDataSource.GetOrCreatePermissionToken"),a=g.decodeFormId(e),i=a.isGroupOwner?"groups":"users",o=a.orgId,s=a.ownerId,c="".concat(this._baseEndpoint,"/formapi/api/").concat(o,"/").concat(i,"/").concat(s,"/forms('").concat(e,"')/Microsoft.FormServices.GetOrCreatePermissionToken"),d=t?new Headers({ShareInvitationKey:t}):void 0;return this._sendRequest("POST",c,{type:"AnalyzerToken",name:"AnalyzerToken"},d).then(function(e){return n.writeSuccess(),e}).catch(function(e){return n.writeUnexpectedFailure("CreateFormFailure",e),Promise.reject(e)})},Object.defineProperty(e.prototype,"endPointUrl",{get:function(){return this._baseEndpoint},enumerable:!1,configurable:!0}),e.prototype.getLongUrlFromShort=function(e){var t=new r._QosMonitor("FormsDataSource.GetLongUrlFromShort");return this._httpClient.fetch("".concat(e,"/uri"),m.HttpClient.configurations.v1,{method:"GET",headers:{Accept:"application/json"}}).then(function(e){return 200===e.status?e.json():(t.writeUnexpectedFailure("ParseResponseError",new Error(e.status.toString())),"")}).catch(function(e){return t.writeUnexpectedFailure("SendRequestFailure",e),Promise.reject(e)})},e.prototype._sendRequest=function(e,t,n,a){var i=this;return this._getOAuthToken().then(function(o){var s=new r._QosMonitor("FormsDataSource.SendRequest"),c=new Headers({Accept:"application/json",Authorization:"Bearer "+o,"Content-Type":"application/json; charset=utf-8","x-ms-form-request-source":"forms-webpart"});return a&&a.forEach(function(e,t){c.append(t,e)}),i._httpClient.fetch(t,m.HttpClient.configurations.v1,{method:e,headers:c,body:JSON.stringify(n)}).then(function(e){return e.status>=200&&e.status<300?e.json():e.text().then(function(e){var t=JSON.parse(e);throw new Error(t.error.code)})}).catch(function(e){return s.writeUnexpectedFailure("SendRequestFailure",e),Promise.reject(e)})})},e.prototype._getOAuthToken=function(){var e=new r._QosMonitor("FormsDataSource.GetOAuthToken");return v()&&!y()?this._aadTokenProvider.getToken(this._baseEndpoint).then(function(t){return e.writeSuccess({alias:"AadTokenProvider"}),t}).catch(function(t){return e.writeUnexpectedFailure("FailedToGetOAuthToken",t),Promise.reject(t)}):this._tokenProvider.getOAuthToken(this._baseEndpoint,this._context.web.serverRelativeUrl).then(function(t){return e.writeSuccess({alias:"LegacyOAuthTokenProvider"}),t.token}).catch(function(t){return e.writeUnexpectedFailure("FailedToGetOAuthToken",t),Promise.reject(t)})},Object.defineProperty(e.prototype,"_baseEndpoint",{get:function(){return"edog"===(this._context.legacyPageContext.env?this._context.legacyPageContext.env:"").toLowerCase()?"https://forms.officeppe.com":"https://forms.office.com"},enumerable:!1,configurable:!0}),e}(),D=(function(){function e(){}e.prototype.createForm=function(e,t){var n={id:"JtSG9haN20KBt6tXjhEMzfH9IyjrPltOvT-NqzY3XrtUMDlYM0dDTzkyTEFZMDIzSk5EUFZTTjlXSCQlQCN0PWcu",title:"Feedback for Forms for Excel",ownerId:"2823fdf1-3eeb-4e5b-bd3f-8dab36375ebb",ownerTenantId:"f686d426-8d16-42db-81b7-ab578e110ccd",xlWorkbookId:"WaitingForExport"};return new Promise(function(e){e(n)})},e.prototype.getAccessToken=function(){return new Promise(function(e){e("access_token")})},e.prototype.getForms=function(e,t,n){var a=[{id:"JtSG9haN20KBt6tXjhEMzfH9IyjrPltOvT-NqzY3XrtUMDlYM0dDTzkyTEFZMDIzSk5EUFZTTjlXSCQlQCN0PWcu",title:"Form 1",ownerId:"2823fdf1-3eeb-4e5b-bd3f-8dab36375ebb",ownerTenantId:"f686d426-8d16-42db-81b7-ab578e110ccd",xlWorkbookId:"WaitingForExport"},{id:"JtSG9haN20KBt6tXjhEMzfH9IyjrPltOvT-NqzY3XrtUMDlYM0dDTzkyTEFZMDIzSk5EUFZTTjlXSCQlQCN0PWcu",title:"Form 2",ownerId:"2823fdf1-3eeb-4e5b-bd3f-8dab36375ebb",ownerTenantId:"f686d426-8d16-42db-81b7-ab578e110ccd",xlWorkbookId:"WaitingForExport"}];return new Promise(function(e){e(a)})},e.prototype.getOrCreatePermissionToken=function(e,t){var n={name:"AnalyzerToken",principalId:"nsYAGqtJevwSTdK5ncVLYKG4FFaqrZsT",type:"Link"};return new Promise(function(e){e(n)})},Object.defineProperty(e.prototype,"endPointUrl",{get:function(){return"https://forms.officeppe.com"},enumerable:!1,configurable:!0}),e.prototype.getLongUrlFromShort=function(e){return new Promise(function(e){e("long url")})}}(),function(){function e(e,t,n){this._dataSource=n||new S(t)}return Object.defineProperty(e.prototype,"endPointUrl",{get:function(){return this._dataSource.endPointUrl},enumerable:!1,configurable:!0}),e.prototype.createForm=function(e,t){return this._dataSource.createForm(e,t)},e.prototype.getAccessToken=function(){return this._dataSource.getAccessToken()},e.prototype.getForms=function(e,t,n){return this._dataSource.getForms(e,t,n)},e.prototype.getOrCreatePermissionToken=function(e,t){return this._dataSource.getOrCreatePermissionToken(e,t)},e.prototype.getDesignPageUrl=function(e){var t=encodeURIComponent("FormId=".concat(e));return d.Uri.concatenate(this._dataSource.endPointUrl,"Pages/DesignPage.aspx?auth_pvr=OrgId&Fragment=".concat(t))},e.prototype.getEmbedResponsePageUrl=function(e){return d.Uri.concatenate(this._dataSource.endPointUrl,"Pages/ResponsePage.aspx?id=".concat(encodeURIComponent(e)))},e.prototype.getEmbedAnalyzeViewUrl=function(e,t){return d.Uri.concatenate(this._dataSource.endPointUrl,"Pages/AnalysisPage.aspx?id=".concat(encodeURIComponent(e),"&AnalyzerToken=").concat(encodeURIComponent(t)))},e.prototype.getLongUrlFromShort=function(e){return this._dataSource.getLongUrlFromShort(e)},e}()),I=(function(){function e(){}Object.defineProperty(e.prototype,"groupId",{get:function(){return"0c9f14ed-cd01-473f-a27a-2e95c79ae634"},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"webTemplate",{get:function(){return"1"},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"currentUICultureName",{get:function(){return"en-us"},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isAnonymousGuestUser",{get:function(){return!1},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isExternalGuestUser",{get:function(){return!1},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"themeColor",{get:function(){return"#e3068b"},enumerable:!1,configurable:!0})}(),function(){function e(e){this._context=e}return Object.defineProperty(e.prototype,"groupId",{get:function(){return this._context.legacyPageContext.groupId},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"webTemplate",{get:function(){return this._context.web.templateName},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"currentUICultureName",{get:function(){return this._context.cultureInfo.currentUICultureName},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isAnonymousGuestUser",{get:function(){return this._context.user.isAnonymousGuestUser},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isExternalGuestUser",{get:function(){return this._context.user.isExternalGuestUser},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"themeColor",{get:function(){return this._context.legacyPageContext.groupColor||this._context.legacyPageContext.siteColor},enumerable:!1,configurable:!0}),e}()),x=function(){function e(e,t,n){this._dataSource=n||new I(t)}return Object.defineProperty(e.prototype,"groupId",{get:function(){return this._dataSource.groupId},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"webTemplate",{get:function(){return this._dataSource.webTemplate},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"currentUICultureName",{get:function(){return this._dataSource.currentUICultureName},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isOnTeamSite",{get:function(){return"64"===this._dataSource.webTemplate},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isOnCommSite",{get:function(){return"68"===this._dataSource.webTemplate},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"hasModernGroup",{get:function(){var e=i.Guid.tryParse(this._dataSource.groupId);return!!e&&e!==i.Guid.empty},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isCurrentUserGuest",{get:function(){return this._dataSource.isAnonymousGuestUser||this._dataSource.isExternalGuestUser},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"themeColor",{get:function(){if(this._dataSource.themeColor)return 0===this._dataSource.themeColor.indexOf("#")?this._dataSource.themeColor.substr(1):this._dataSource.themeColor},enumerable:!1,configurable:!0}),e}();n("jOlS").loadStyles(".a_a_9931711e{height:640px}.a_a_9931711e iframe{border:none;height:100%;width:100%}.b_a_9931711e{color:#0078d4;display:none;height:0;width:0}",!0);var C=function(e){function t(t,n){var a=e.call(this,t,n)||this;return a._formsEmbedContainerId="formsEmbedContainer_".concat(a.props.webPartId),a._formsEmbedIframeId="formsEmbedIframe_".concat(a.props.webPartId),a._hostId=a.props.webPartId.replace(/-/g,""),a._formsEmbedContainerDivRef=f.createRef(),a._formsThemeColorDivRef=f.createRef(),a}return Object(a.__extends)(t,e),t.prototype.render=function(){return f.createElement("div",null," ",!this._shouldShowPlaceholder&&f.createElement("div",{id:this._formsEmbedContainerId,className:"a_a_9931711e",ref:this._formsEmbedContainerDivRef,tabIndex:0,"aria-label":this._getAccessibilityLabel()})," ",this._shouldShowPlaceholder&&this.props.isEdit&&f.createElement("div",{"data-automation-id":"forms-webpart-container"},f.createElement(l.Placeholder,{icon:"OfficeFormsLogo",iconText:b.l,description:b.d,buttonLabel:b.c,onAdd:this._showCreateFormsPropertyPane,extraButtons:[{label:b.f,onClick:this._showInsertFormsPropertyPane,description:b.u}]}))," ",f.createElement("div",{className:"b_a_9931711e",ref:this._formsThemeColorDivRef}))},t.prototype.componentDidMount=function(){window.addEventListener("message",this._onReceiveFormsMessage,!1),this._shouldShowPlaceholder||this._loadFormsResult(),this._removeDupMainRole()},t.prototype.componentDidUpdate=function(e){(this.props.shouldRefreshContainer||e.isEdit!==this.props.isEdit)&&this._loadFormsResult(),this._removeDupMainRole()},t.prototype.componentWillUnmount=function(){window.removeEventListener("message",this._onReceiveFormsMessage,!1)},Object.defineProperty(t.prototype,"_shouldShowPlaceholder",{get:function(){return!this.props.formURL},enumerable:!1,configurable:!0}),t.prototype._getAccessibilityLabel=function(e,t){if(this._shouldShowPlaceholder)return b.n;var n=e?d.StringHelper.format(b.r,decodeURIComponent(e)):"",a=t&&1===this.props.displayType?d.StringHelper.format(b.a,decodeURIComponent(t)):"",i=2===this.props.displayType?b.s:b.o;return d.StringHelper.format(b.i,n,a,i)},Object.defineProperty(t.prototype,"_themeColor",{get:function(){return this._primaryThemeColor||(this._primaryThemeColor=g.getFormThemeColor(this._formsThemeColorDivRef.current)),this._primaryThemeColor},enumerable:!1,configurable:!0}),t.prototype._loadFormsResult=function(e){var n=this;for(void 0===e&&(e=!1),this._qosMonitor=new r._QosMonitor("FormsEmbedContainer.RenderForms"),this._async.setTimeout(function(){return n._endQosMonitor(!1)},t._qosEventTimeout);this._formsEmbedContainerDivRef.current.firstChild;)this._formsEmbedContainerDivRef.current.removeChild(this._formsEmbedContainerDivRef.current.firstChild);this._formsIFrame=document.createElement("iframe"),this._formsIFrame.id=i._SPKillSwitch.isActivated("69d31729-0973-4716-a9eb-e02a20b16b40")?this._formsEmbedContainerId:this._formsEmbedIframeId,this._formsIFrame.tabIndex=0,this._formsIFrame.setAttribute("aria-label",this._getAccessibilityLabel());var a=new d.Uri(this.props.formURL);a.setQueryParameter("lang",this.props.cultureName),a.setQueryParameter("themecolor",this._themeColor),a.setQueryParameter("oembedsso","true"),a.setQueryParameter("hostId",this._hostId),a.setQueryParameter("origin",t._logEventPrefix),this.props.isEdit&&1===this.props.displayType&&a.setQueryParameter("preview","true"),this._formsIFrame.src=a.toString(),this._formsEmbedContainerDivRef.current.appendChild(this._formsIFrame)},t.prototype._onReceiveFormsMessage=function(e){var t,n=this;if(e.data&&e.origin===this.props.endPointUrl){var a=e.data.toLowerCase();if(a.indexOf("getaccesstokenfromhost")>=0)this.props.getAccessTokenHandler().then(function(t){e.source.postMessage("FormsAccessTokenFromHost:Bearer ".concat(t),n.props.endPointUrl)}).catch(function(e){return n.props.oauthErrorRenderer(e)});else if(a.indexOf("formspageheight")>=0){if(3===(d=e.data.split(":")).length&&d[1]===this._hostId){var i=Number(d[2]),r="".concat(i=i<408?408:i>3480?3480:i,"px");(null===(t=this._formsEmbedContainerDivRef.current)||void 0===t?void 0:t.style.height)!==r&&(this._formsEmbedContainerDivRef.current.style.height=r)}}else if(a.indexOf("formsinfo")>=0){if(4===(d=e.data.split(":")).length&&d[1]===this._hostId){this._endQosMonitor(!0);var o=decodeURIComponent(d[2]),s=decodeURIComponent(d[3]),c=this._getAccessibilityLabel(o,s);this._formsEmbedContainerDivRef.current.setAttribute("aria-label",c),this._formsIFrame.setAttribute("aria-label",o)}}else if(a.indexOf("formserror")>=0){var d;if(3===(d=e.data.split(":")).length&&d[1]===this._hostId){var l=new Error(d[2]);this._endQosMonitor(!1,l)}}}},t.prototype._showInsertFormsPropertyPane=function(){r._EngagementLogger.logEvent("".concat(t._logEventPrefix,".AddExisting.Click")),this.props.openPropertyPaneCallback(2)},t.prototype._showCreateFormsPropertyPane=function(){r._EngagementLogger.logEvent("".concat(t._logEventPrefix,".CreateNew.Click")),this.props.openPropertyPaneCallback(1)},t.prototype._endQosMonitor=function(e,t){this._qosMonitor&&!this._qosMonitor.hasEnded&&(e?this._qosMonitor.writeSuccess():t?this._qosMonitor.writeExpectedFailure("RenderError",t):this._qosMonitor.writeUnexpectedFailure("RenderTimeout"))},t.prototype._removeDupMainRole=function(){if(!i._SPKillSwitch.isActivated("e5b54a26-aee6-4403-bf8b-29cbddc3145f")){var e=document.querySelector('#form-main-content1 div div[role="main"]');null!==e&&e.setAttribute("role","")}},t._logEventPrefix="FormsWebPart",t._qosEventTimeout=12e4,Object(a.__decorate)([u.autobind],t.prototype,"_onReceiveFormsMessage",null),Object(a.__decorate)([u.autobind],t.prototype,"_showInsertFormsPropertyPane",null),Object(a.__decorate)([u.autobind],t.prototype,"_showCreateFormsPropertyPane",null),Object(a.__decorate)([u.autobind],t.prototype,"_endQosMonitor",null),Object(a.__decorate)([u.autobind],t.prototype,"_removeDupMainRole",null),t}(u.BaseComponent),O=function(){function e(e,t,n,a){this._properties=e,this._onClickCreateButton=t,this._onClickInsertButton=n,this._formsEndPointURL=a}return e.prototype.getPropertyPanePages=function(e,t,n){e&&(this._properties=e);var a=[{header:{description:n?b.g:b.p},groups:[{groupFields:[this._createFormsTitleInputBox(),this._createLearnMoreLink()]},{groupFields:[this._createFormsCreateButton()]}]}],i=[{groups:[{groupFields:this._properties.formURL&&g.isValidFormUrl(this._properties.formDesignPageURL)?[this._createEditFormsLink()]:[this._createGoToFormsLabel(),this._createGoToFormsLink()]},{groupFields:[this._createResultLinkInputBox(),this._createLearnMoreLink()]},{groupFields:[this._createDisplayChoiceGroup()]},{groupFields:2!==this._properties.displayType?[]:[this._createShowResultCheckBox()]},{groupFields:[this._createFormsInsertButton()]}]}];return t?{pages:a}:{pages:i}},e.prototype._createLearnMoreLink=function(){return Object(s.PropertyPaneLink)("learnMore",{ariaLabel:b.O,text:b.w,target:"_blank",href:"https://go.microsoft.com/fwlink/?linkid=854172"})},e.prototype._createResultLinkInputBox=function(){return Object(s.PropertyPaneTextField)("inputURL",{label:b.x,multiline:!0,placeholder:"".concat(this._formsEndPointURL,"/Pages/…"),value:this._properties.inputURL,errorMessage:this._properties.errorMessage})},e.prototype._createFormsTitleInputBox=function(){return Object(s.PropertyPaneTextField)("formTitle",{label:b.h,multiline:!1,placeholder:b.b,errorMessage:this._properties.errorMessage})},e.prototype._createFormsCreateButton=function(){var e=this;return Object(s.PropertyPaneButton)("createForms",{ariaLabel:b.v,text:b.y,onClick:function(){return e._onClickCreateButton()}})},e.prototype._createFormsInsertButton=function(){var e=this;return Object(s.PropertyPaneButton)("insertForms",{text:b.C,onClick:function(){return e._onClickInsertButton()},disabled:!this._properties.allowShowResult&&2===this._properties.displayType})},e.prototype._createDisplayChoiceGroup=function(){return Object(s.PropertyPaneChoiceGroup)("displayType",{options:[{key:1,text:b._},{key:2,text:b.m}]})},e.prototype._createShowResultCheckBox=function(){return Object(s.PropertyPaneCheckbox)("allowShowResult",{text:b.k})},e.prototype._createGoToFormsLabel=function(){return Object(s.PropertyPaneLabel)("goToFormsLabel",{text:b.D})},e.prototype._createGoToFormsLink=function(){return Object(s.PropertyPaneLink)("goToForms",{ariaLabel:b.I,text:b.I,target:"_blank",href:this._formsEndPointURL})},e.prototype._createEditFormsLink=function(){return Object(s.PropertyPaneLink)("editForms",{ariaLabel:b.S,text:b.S,target:"_blank",href:this._properties.formDesignPageURL})},e}(),w=function(e){function t(){return null!==e&&e.apply(this,arguments)||this}return Object(a.__extends)(t,e),t.prototype.render=function(){this._oAuthError?this._renderOAuthErrorRenderer(this._oAuthError):this._renderFormsWebPart()},t.prototype.onInit=function(){var t=this;return this._formsDataProvider=this._formsDataProvider||new D(i.Environment.type,this.context.serviceScope),this._sharePointDataProvider=this._sharePointDataProvider||new x(i.Environment.type,this.context.pageContext),this.properties.displayType=g.getDisplayTypeFromFormUrl(this.properties.formURL),this.properties.allowShowResult=!0,this._propertyPane=new O(this.properties,this._handleCreateForms,this._handleEmbedForms,this._formsDataProvider.endPointUrl),this._handleGetAccessToken().catch(function(e){return t._oAuthError=e}).then(e.prototype.onInit)},Object.defineProperty(t.prototype,"dataVersion",{get:function(){return i.Version.parse("1.0")},enumerable:!1,configurable:!0}),Object.defineProperty(t.prototype,"propertiesMetadata",{get:function(){return{formDesignPageURL:{isLink:!0},formTitle:{isSearchablePlainText:!0},formURL:{isLink:!0},inputURL:{isLink:!0}}},enumerable:!1,configurable:!0}),t.prototype.getPropertyPaneConfiguration=function(){var e=this.context.propertyPane.isRenderedByWebPart()&&1===this._actionType;return this._propertyPane.getPropertyPanePages(this.properties,e,this._sharePointDataProvider.hasModernGroup)},t.prototype.onPropertyPaneFieldChanged=function(e,t,n){"insertForms"!==e&&"createForms"!==e&&(this.properties.errorMessage="",this._shouldRefreshContainer=!1)},t.prototype.onAfterResize=function(e){i._SPKillSwitch.isActivated(i.Guid.parse("c40ee8a3-2544-4514-bd70-70772b7b100d"),"08/29/2017","NoRefreshFormsContainer")||(this._shouldRefreshContainer=!1),this.render()},t.prototype.onDispose=function(){p.unmountComponentAtNode(this.domElement),e.prototype.onDispose.call(this)},t.prototype._renderFormsWebPart=function(){var e=this,t=Object(a.__assign)({actionType:this._actionType,cultureName:this._sharePointDataProvider.currentUICultureName,endPointUrl:this._formsDataProvider.endPointUrl,themeColor:this._sharePointDataProvider.themeColor,getAccessTokenHandler:this._handleGetAccessToken,oauthErrorRenderer:this._renderOAuthErrorRenderer,isEdit:this.displayMode===i.DisplayMode.Edit,openPropertyPaneCallback:function(t){e._actionType=t,e.properties.errorMessage="",e.context.propertyPane.open()},shouldRefreshContainer:!!this._shouldRefreshContainer,webPartId:this.context.instanceId},this.properties);p.render(f.createElement(C,t),this.domElement)},t.prototype._renderOAuthErrorRenderer=function(e){var t={error:e,customErrorMessage:e.redirectUrl?void 0:b.e,webpartName:b.l,webpartAlias:this.context.manifest.alias,webpartIcon:"OfficeFormLogo"},n=f.createElement(l.OAuthErrorRenderer,t);p.render(n,this.domElement)},t.prototype._handleCreateForms=function(){var e=this,n=new o._LogEntry(t._moduleName,"CreateForms",o._LogType.Event,{siteType:this._sharePointDataProvider.hasModernGroup?"groupSite":"commsSite"});if(r._EngagementLogger.logEventWithLogEntry(n),g.isValidTitle(this.properties.formTitle)){var a=d.StringHelper.format(b.t,this.properties.formTitle);c.ScreenReader.alert("NewFormAlerts",a);var i=window.open(),s={title:this.properties.formTitle,ownerTenantId:this.context.pageContext.aadInfo.tenantId.toString(),ownerId:this._sharePointDataProvider.groupId||this.context.pageContext.aadInfo.userId.toString(),xlWorkbookId:"WaitingForExport",settings:JSON.stringify({IsAnonymous:!1,IsQuizMode:!1,RequiresUniqueResponse:!1,ShowGradedScores:!1})},l=this._sharePointDataProvider.hasModernGroup;this._formsDataProvider.createForm(s,l).then(function(t){var n=t.id;e.properties.formDesignPageURL=e._formsDataProvider.getDesignPageUrl(n),e.properties.formURL=e._formsDataProvider.getEmbedResponsePageUrl(n),e.properties.inputURL=e._formsDataProvider.getDesignPageUrl(n),i.location.href=e.properties.inputURL,e.properties.displayType=1,e._actionType=2,e.context.propertyPane.refresh(),e._shouldRefreshContainer=!0,e._renderFormsWebPart()}).catch(this._showPropertyPaneErrorMessage)}else this.properties.errorMessage=b.U},t.prototype._handleEmbedForms=function(){var e=this;!i._SPKillSwitch.isActivated("06f5f348-a31e-49d0-85f6-b662967f301e")&&g.isValidShortUrlFormat(this.properties.inputURL)?this._formsDataProvider.getLongUrlFromShort(this.properties.inputURL).then(function(t){e.properties.inputURL=t,e._handleEmbedFormsInner(!0)}).catch(function(t){e._handleEmbedFormsInner()}):this._handleEmbedFormsInner()},t.prototype._handleEmbedFormsInner=function(e){void 0===e&&(e=!1);var n=new o._LogEntry(t._moduleName,"EmbedForms",o._LogType.Event,{displayType:2===this.properties.displayType?"showResult":"showSurvey",siteType:this._sharePointDataProvider.hasModernGroup?"groupSite":"commsSite",pastedURLType:g.pastedURLType(this.properties.inputURL)});r._EngagementLogger.logEventWithLogEntry(n);var a=g.getFormIdFromUrl(this.properties.inputURL);this.properties.errorMessage=g.getEmptyUrlErrorMessage(this.properties.inputURL)||g.getIncorrectDomainErrorMessage(this.properties.inputURL)||g.getInvalidValidFormIdErrorMessage(a),this.properties.errorMessage||this._tryGenerateEmbedFormUrl(a,e)||(this.properties.errorMessage=b.P)},t.prototype._handleGetAccessToken=function(){return this._formsDataProvider.getAccessToken()},t.prototype._tryGenerateEmbedFormUrl=function(e,t){var n=this;if(2===this.properties.displayType){if(2===g.getDisplayTypeFromFormUrl(this.properties.inputURL))this.properties.formURL=this.properties.inputURL,this.properties.formDesignPageURL=this._formsDataProvider.getDesignPageUrl(e),this._shouldRefreshContainer=!0;else{var a=g.getCoAuthTokenFromFormUrl(this.properties.inputURL);this._formsDataProvider.getOrCreatePermissionToken(e,a).then(function(t){n.properties.formURL=n._formsDataProvider.getEmbedAnalyzeViewUrl(e,t.principalId),n.properties.formDesignPageURL=n._formsDataProvider.getDesignPageUrl(e),n._shouldRefreshContainer=!0,n.context.propertyPane.refresh(),n._renderFormsWebPart()}).catch(this._showPropertyPaneErrorMessage)}return!0}return 1===this.properties.displayType&&(this.properties.formURL=this._formsDataProvider.getEmbedResponsePageUrl(e),this.properties.formDesignPageURL=this._formsDataProvider.getDesignPageUrl(e),this._shouldRefreshContainer=!0,t&&(this.context.propertyPane.refresh(),this._renderFormsWebPart()),!0)},t.prototype._showPropertyPaneErrorMessage=function(e){this.properties.errorMessage=g.getErrorMessageFromError(e),this.context.propertyPane.refresh()},t._moduleName="FormsWebPart",Object(a.__decorate)([u.autobind],t.prototype,"_renderFormsWebPart",null),Object(a.__decorate)([u.autobind],t.prototype,"_renderOAuthErrorRenderer",null),Object(a.__decorate)([u.autobind],t.prototype,"_handleCreateForms",null),Object(a.__decorate)([u.autobind],t.prototype,"_handleEmbedForms",null),Object(a.__decorate)([u.autobind],t.prototype,"_handleGetAccessToken",null),Object(a.__decorate)([u.autobind],t.prototype,"_showPropertyPaneErrorMessage",null),t}(s.BaseClientSideWebPart);t.default=w},U4ag:function(e,t){e.exports=o},UWqr:function(e,t){e.exports=s},"X+PM":function(e,t){e.exports=c},br4S:function(e,t){e.exports=d},cDcd:function(e,t){e.exports=l},faye:function(e,t){e.exports=u},jOlS:function(e,t){e.exports=f},mGD9:function(e){e.exports=JSON.parse('{"e":"Can\\u0027t access the Forms.","t":"A new browser tab is opening. You will land in Microsoft Forms {0} page.","n":"Microsoft Forms","i":"Microsoft Forms. {0} {1} {2}","r":"Form title: {0}.","a":"Form description: {0}.","o":"Press enter key to enter the form content, press tab key to navigate inside the form.","s":"Use Caps Lock plus right arrow to navigate inside the form.","c":"New form","d":"Easily create surveys, quizzes, and polls","l":"Microsoft Forms","u":"Insert a form","f":"Add existing form","p":"Create a new form in Microsoft Forms, and then come back to this tab.","m":"Show form results","_":"Collect responses","h":"New form","b":"Name your form","g":"Create a new form in Microsoft Forms, and then come back to this tab.","v":"Create button. Upon clicking create button, a new form will be created and opened in a new browser tab.","y":"Create","S":"Edit current form","D":"Collect form responses or share results from Microsoft Forms by pasting your form\\u0027s web address below.","I":"Go to Microsoft Forms","x":"Form web address","C":"OK","w":"Learn more","O":"Link in the Forms settings panel that connects to set up guidance.","k":"A web address will be created. Anyone with it can view a summary of responses.","M":"Form Id is invalid. Please try to copy a form from Microsoft Forms.","P":"Something went wrong. Please try again.","T":"The URL cannot be empty. Please copy a URL from Microsoft Forms.","U":"The title\\u0027s length is out of range. Please try to input title with less than 90 characters.","F":"This is not a valid form link. Please copy a URL from Microsoft Forms.","E":"Your account is not enabled for Microsoft Forms.","L":"This form does not have share permissions.","A":"To embed a summary, the owner of this form needs to change collaboration permissions to an option other than \\u0022Specific people in my organization can view and edit\\u0022."}')},ut3N:function(e,t){e.exports=p},vlQI:function(e,t){e.exports=m},y88i:function(e,t){e.exports=_},ytfe:function(e,t){e.exports=h}})});
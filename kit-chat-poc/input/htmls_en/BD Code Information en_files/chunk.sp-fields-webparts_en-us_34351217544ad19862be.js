(window.webpackJsonp_4ec0946ccf7b5264836bb62352873336=window.webpackJsonp_4ec0946ccf7b5264836bb62352873336||[]).push([[8],{KZze:function(e,t,n){"use strict";var a;n.d(t,"e",function(){return a}),function(e){e[e.Normal=0]="Normal",e[e.Divider=1]="Divider",e[e.Header=2]="Header",e[e.SelectAll=3]="SelectAll"}(a||(a={}))},MxUo:function(e,t,n){"use strict";n.r(t);var a=n("17wl"),i=n("b69B"),r=n("JyUB"),o=n("dq6a"),s=n("K9kD"),c="__next__",d=s.Killswitch.isActivated("4D69327F-16ED-4A50-8ABF-F476ABC4E29B","05/04/2023","should only retry request when groupReplace was not true in the context"),l=s.Killswitch.isActivated("23DCF7B8-CE78-43EA-8F10-7B5CD47411D5","05/04/2023","return retriedWithGroupReplaceContext as part of listItemResponse"),u=s.Killswitch.isActivated("270dfbb4-1b85-4fed-9ada-b85e650353f4","6/9/22","retry request for large list"),f=s.Killswitch.isActivated("3f7e1f54-43e3-4c37-a5f3-b78fb07ed315","7/12/22","read groupString from postDataContext when listContext group is empty"),p=s.Killswitch.isActivated("c49d1ed9-cfb5-4a27-ab34-801c001bcb9f","4/12/2023"),m=function(e){function t(t){return e.call(this,{dataSourceName:"ListItemDataSource"},{pageContext:t.pageContext})||this}return Object(a.__extends)(t,e),t.prototype.getItem=function(t,n,a,i,r,o){var c=this;t.postDataContext.isOnePage=!!t.newTargetListUrl||!!t.isOnePageNavigation;var l=t.remoteItem?this._getRemoteItemPostData():this.getAdditionalPostData(t.postDataContext,n,t.pageSize),f=function(){return t.remoteItem?c._getRemoteItemPostData():c.getAdditionalPostData(t.postDataContext,n,t.pageSize)};return(!p&&this.getUrlAsync?this.getUrlAsync(n,t):s.Promise.resolve(void 0)).then(function(p){var m=function(d){return e.prototype.getData.call(c,p?function(){return p}:function(){return c.getUrl(n,t)},function(e,t){return c._parseResponse(e,a.qosEvent,t,d)},a.qosName,u?function(){return l}:f,"POST",c._getListRequestHeaders(t,n,a),void 0,void 0,t.noRedirect,void 0,function(e){var t=c._telemetryHandler(e);return i&&i(t.eventData),t},void 0,t.authToken,void 0,r,o).then(function(e){return e},function(e){return c._errorHandler(a.qosName,e),s.Promise.reject(e)})};return m().then(function(e){return e},function(e){var n;return u||-1===(null===(n=e.code)||void 0===n?void 0:n.indexOf("Microsoft.SharePoint.SPQueryThrottledException"))||!d&&t.postDataContext.groupReplace?s.Promise.reject(e):(t.postDataContext.groupReplace=!0,m(!0))})})},t.prototype.getUrl=function(e,t){var n,a=t&&t.postDataContext;return n={webUrl:this._pageContext.webAbsoluteUrl,listId:t.remoteItem?t.remoteItem.ListId:e.listId,urlParts:t.remoteItem?{}:e.urlParts,searchTerm:e.searchTerm,rootFolder:t.remoteItem?void 0:e.folderPath,isOnePageNavigation:t.isOnePageNavigation,remoteWebUrl:t.remoteItem?t.remoteItem.SiteUrl:t.remoteWebUrl,uniqueId:t.remoteItem&&t.remoteItem.UniqueId,authToken:null==t?void 0:t.authToken,inplaceSearchMultiLineFieldQuery:e.inplaceSearchMultiLineFieldQuery},s.Killswitch.isActivated("03784CEA-7833-4740-8D6A-4EA05116664F","5/18/21","Set prefetch param")||(n.isPrefetchPageContext=e.listId&&!this._pageContext.listUrl),this._isSpecifiedItemRequest(e)?n.viewId=e.viewIdForRequest:((!e.viewXmlForRequest||a&&a.isOnePage)&&e.viewIdForRequest!==s.Guid.Empty&&(n.view=e.viewIdForRequest),!n.view&&e.viewPathForRequest&&(n.viewPath=e.viewPathForRequest),t&&t.ignoreFilterParams||(n.filterParams=e.filterParams),e.sortField&&(n.sortField=e.sortField,n.sortDir="false"===e.isAscending?"Desc":"Asc"),a&&a.groupReplace?n.requestToken=this._isExpandingGroup(n.groupString)?void 0:e.requestToken:n.requestToken=e.group?void 0:e.requestToken,!e.group||!this._isExpandingGroup(e.group.groupString)||a&&a.groupReplace?f||!t.group||!this._isExpandingGroup(t.group.groupString)||a&&a.groupReplace||(n.groupString=t.group.groupString):n.groupString=e.group.groupString,a&&a.webId&&(n.webId=a.webId)),i.t(n)},t.prototype.getAdditionalPostData=function(e,t,n){return r.e(e,t,n)},t.prototype.needsRequestDigest=function(e){return!!s.Killswitch.isActivated("4CE25DD5-0B47-4B9E-B6F9-5037A00C6CF8","04/21/2021","need Request Digest")||-1===e.toLowerCase().indexOf("/renderlistdataasstream?")},t.prototype._getRemoteItemPostData=function(){return JSON.stringify({parameters:{RenderOptions:7}})},t.prototype._getListRequestHeaders=function(e,t,n){var a=i.n(t,e.postDataContext);return n&&n.application&&(a.Application=n.application,a.Scenario="ViewList",a.ScenarioType="AUO"),a["X-ServiceWorker-Strategy"]=e.skipServiceWorkerCache?"SkipCache":"CacheFirst",a},t.prototype._isSpecifiedItemRequest=function(e){return e.itemIds&&e.itemIds.length>0},t.prototype._isExpandingGroup=function(e){return e&&e!==c},t.prototype._parseResponse=function(e,t,n,a){var i=n.getServiceWorkerDataSourceHeader();try{if(void 0!==e)return this._getListItemResponse(e,i,a)}catch(n){if("{}"!==e.substring(0,2))throw t.end({resultType:s.ResultTypeEnum.Failure,resultCode:"BadJSON"}),n;return this._getListItemResponse(e.substring(2),i,a)}},t.prototype._getListItemResponse=function(e,t,n){var a;return a=JSON.parse(e),"ServiceWorker-FromCache"===t&&(a.isFromServiceWorkerCache=!0),!l&&n&&(a.retriedWithGroupReplaceContext=!0),a},t.prototype._telemetryHandler=function(e){var t=e.errorData,n=Object(a.__assign)({},e.eventData);if(t){var i=void 0,r=t.code?t.code:"";return r.indexOf("2147024860")>-1?r="ListViewTreshold":r.indexOf("2147024749")>-1?r="LookupColumnTreshold":r.indexOf("2147024809")>-1&&r.indexOf("requestUrl")>-1?r="RequestURLFailure":r.indexOf("2147024809")>-1&&r.indexOf("range")>-1?r="RangeError":r.indexOf("2147024809")>-1&&r.indexOf("view")>-1?r="InvalidView":r.indexOf("2130575340")>-1&&r.indexOf("field")>-1?r="FieldTypesNotInstalledProperly":r.indexOf("2146232832")>-1?(r="DefaultDoclibNotFound",500===t.status&&(i=!1,n.resultType=s.ResultTypeEnum.ExpectedFailure)):r.indexOf("2147024895")>-1?(r="PrefetchWebMismatch",500===t.status&&(i=!1,n.resultType=s.ResultTypeEnum.ExpectedFailure)):r=n.resultCode,n.resultCode=r,{canRetry:i,eventData:n}}return{eventData:n}},t.prototype._errorHandler=function(e,t){var n=t&&t.code?t.code:"";n.indexOf("2147024860")>-1?s.Engagement.logData({name:e+".ListViewTreshold"}):n.indexOf("2147024749")>-1&&s.Engagement.logData({name:e+".LookupColumnTreshold"})},t}(o.t),_=n("7Xy9"),h=n("ftiL"),b=n("xkft"),g=function(){function e(e){this._pageContext=e.pageContext,this._retriever=new m({pageContext:this._pageContext}),this._itemStore=e.itemStore,this._itemUrlHelper=e.itemUrlHelper||new b.e({},{pageContext:this._pageContext})}return e.prototype.getItem=function(e,t){var n=this,a={qosEvent:new s.Qos({name:"GetListViewData"}),qosName:"GetListViewData"};return function(e,t){var n=t.group&&t.group.groupingId===c,a=!!e.needSchema||!t.listSchema||n,i=!t.viewResult,o=void 0;if(n)o=r.t({sortField:t.sortField,itemIds:void 0,isAscending:t.isAscending,pageSize:e.pageSize||100,fetchNextGroup:n,lastGroup:t.lastGroup,recurseInFolders:!1,fieldNames:void 0,typeFilter:void 0,groupBy:t.groupByOverride?[t.groupByOverride]:t.groupBy,userIsAnonymous:!1,requestMetaInfo:!1});else if(e.viewXml&&(o=e.viewXml,t.sortField||t.groupByOverride)){var s=new h.e(o);t.sortField&&s.updateSort(null,{overwriteAll:!0}),t.groupByOverride&&s.updateGroupBy({isCollapsed:!1,group1:{fieldName:t.groupByOverride}}),o=s.getEffectiveViewXml()}e.postDataContext={needsSchema:a,needsContentTypes:!0,needsContentTypeOrder:!1,needsForms:!1,needsQuickLaunch:!1,needsSpotlight:!1,needsViewMetadata:i,needsParentInfo:!1,viewXml:o,firstGroupOnly:!1,expandGroups:e.expandGroup,allowMultipleValueFilterForTaxonomyFields:!1,requestToken:e.requestToken,fieldNames:void 0,isListDataRenderOptionChangeFeatureEnabled:!0,isSpotlightFeatureEnabled:!1,groupByOverride:e.groupBy,requestDatesInUtc:!1,needClientSideComponentManifest:!1}}(e,t),this._retriever.getItem(e,t,a).then(function(a){return n._fixupContextIfNeeded(e,t,a),new _.e({listContext:t,pageContext:n._pageContext,itemStore:n._itemStore}).processData(e.parentKey,a)},function(e){return s.Promise.reject(e)})},e.prototype._fixupContextIfNeeded=function(e,t,n){if(!e.parentKey){var a=decodeURIComponent(n.listUrlDir);t.urlParts=this._itemUrlHelper.getUrlParts({listUrl:a}),e.parentKey=a}},e}(),v=n("Rg6q"),y=n("IV9z"),S=function(){function e(e){var t=!1;if(e.listContext&&e.pageContext&&(this._listContext=e.listContext,this._pageContext=e.pageContext,this._listItemDataSource=new g({pageContext:this._pageContext}),this._clientFormProvider=new y.e({pageContext:this._pageContext,listContext:this._listContext}),t=!!this._listItemDataSource&&!!this._clientFormProvider),!t)throw new Error("Invalid param for PropertiesClientFormProvider")}return e.prototype.getClientForm=function(e){var t=this;return Promise.resolve().then(function(){if(e&&!(e.length<=0)){var n={clientFormType:v.e.editItem,itemId:"".concat(e[0]),isBulkEdit:e.length>1};return t._clientFormProvider.getClientForm(n)}})},e.prototype.getClientFormSchema=function(){return this._clientFormProvider._clientFormDataSource.getClientFormSchema(v.e.editItem)},e.prototype.updateItem=function(e,t){var n={clientForm:e,newDocumentUpdate:!1,isInteractiveSave:t.length<=1,itemdIds:t};return this._clientFormProvider.updateItem(n)},e.prototype.getClientFormProvider=function(){return this._clientFormProvider},e}(),D=function(){function e(){}return e.prototype.createClientFormProvider=function(e){return new S(e)},e}();t.default=D}}]);
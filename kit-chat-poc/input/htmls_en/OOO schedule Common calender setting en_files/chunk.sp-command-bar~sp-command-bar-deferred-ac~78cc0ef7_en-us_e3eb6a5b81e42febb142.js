(window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0=window.webpackJsonpb6917cb1_93a0_4b97_a84d_7cf49975d4ec_0_2_0||[]).push([[6],{"2G27":function(e,t,n){"use strict";var a=n("DGFG");t.e=function(e){this.name="LikeLimitExceeded",this.message=a.n,e&&(this.message=e)}},"D/Gr":function(e,t,n){"use strict";var a=n("Rrln"),i=n("i8gT"),r=function(){function e(e,t,n,a,i,r,o,s,c,d,l,u,f){this._isDeleted=!1,this._date=t,this._id=n,this._nextLink=a,this._parentId=i,this._text=r,this._replies=e,this._author=o,this._likeCount=s,this._userLiked=c,this._likedBy=d,this._nextLikesLink=l,this._serverObject=u,this._mentions=f,u||(this._optimisticId=n),this.sortReplies()}return Object.defineProperty(e.prototype,"type",{get:function(){return"Comment"},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"date",{get:function(){return this._date},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"id",{get:function(){return this._id},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"optimisticId",{get:function(){return this._optimisticId},set:function(e){this._optimisticId=e},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"isFromServer",{get:function(){return!!this._serverObject},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"nextLink",{get:function(){return this._nextLink},set:function(e){this._nextLink=e},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"parentId",{get:function(){return this._parentId},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"text",{get:function(){return this._text},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"replies",{get:function(){return this._replies},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"author",{get:function(){return this._author},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"likeCount",{get:function(){return this._likeCount},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"likedBy",{get:function(){return this.userLiked?[a.e.instance.user].concat(this._likedBy||[]):this._likedBy},set:function(e){this._likedBy=e},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"nextLikesLink",{get:function(){return this._nextLikesLink},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"mentions",{get:function(){return this._mentions},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"userLiked",{get:function(){return this._userLiked},enumerable:!1,configurable:!0}),e.prototype.addReply=function(e){this._replies.push(e),this._latestDate=void 0},e.prototype.equals=function(e){return this.id===e.id},e.prototype.hasMoreLikes=function(){return Boolean(this._nextLikesLink)},e.prototype.isDeleted=function(){return this._isDeleted},e.prototype.isReply=function(){return"0"!==this._parentId},e.prototype.latestDate=function(){return this._latestDate||(this._latestDate=this._replies&&this._replies.length?this._replies[this._replies.length-1].date:this._date)},e.prototype.markDeleted=function(){this._isDeleted=!0},e.prototype.moreReplies=function(){return!!this._nextLink},e.prototype.processLikeChunk=function(e){var t=this;this._nextLikesLink=e["@odata.nextLink"],i.e.allLikesFetched(this._nextLikesLink)&&(this._nextLikesLink=void 0);var n=e.likes||[];this.userLiked&&(n=n.filter(function(e){return!a.e.isCurrentUser(e)})),n=n.filter(function(e){return!t._alreadyLiked(e)}),this._likedBy=(this._likedBy||[]).concat(n)},e.prototype.removeReply=function(e){this._replies=this._replies.filter(function(t){return!e.equals(t)}),this._latestDate=void 0},e.prototype.replaceReply=function(e,t){var n=this._replies.indexOf(e);n>-1?this._replies[n]=t:this.addReply(t)},e.prototype.sortReplies=function(){this._replies=this._replies.sort(function(e,t){return+e.latestDate()-+t.latestDate()}),this._replies=this._replies.filter(function(e,t,n){return 0===t||e.id!==n[t-1].id})},e.prototype.like=function(e){this.userLiked||(this._userLiked=!0,this._likeCount++)},e.prototype.unlike=function(e){this.userLiked&&(this._userLiked=!1,this._likeCount--)},e.prototype.unmarkDelted=function(){this._isDeleted=!1},e.prototype._alreadyLiked=function(e){return this._likedBy.some(function(t){return t.email===e.email})},e}();t.e=r},F5rs:function(e){e.exports=JSON.parse('{"n":"Comments","f":"Comment removed with no replies||Comment removed with {0} reply||Comment removed with {0} replies ","u":"0||1||2- ","o":"No more comments to show||Showing {0} older comment||Showing {0} older comments","s":"0||1||2-","c":"No more replies to show||Showing {0} more reply||Showing {0} more replies","r":"0||1||2-","i":"No more likes to show.||Showing {0} more like.||Showing {0} more likes.","d":"0||1||2-","l":"Comment posted","a":"Comment liked.","p":"Comment unliked.","b":"Reply","g":"Reply to {0}","m":"Delete","_":"Like the comment.","y":"Unlike the comment.","v":"Show more.","h":"More options","I":"Delete confirmation","D":"Are you sure you want to delete this comment?","x":"Couldn\\u0027t delete comment. Try again.","C":"Comments||{0} Comment||{0} Comments","O":"0||1||2-","Y":"Add a comment. Type @ to mention someone","q":"Add a comment.","L":"Post","k":"Couldn\\u0027t post comment. Try again.","w":"Couldn\\u0027t like the comment. Try again.","j":"Couldn\\u0027t unlike the comment. Try again.","E":" Could not like the comment.  The comment has reached the maximum number of likes. ","A":"Likes","S":"Close","F":"Show more","M":"You have exceeded the limit by {0} character||You have exceeded the limit by {0} characters","P":"1||2-","T":"{0} character left||{0} characters left","U":"1||0,2-","H":"Comments on","R":"Comments off","N":"Off","B":"On","V":"Photo of {0}","z":"Show more comments","K":"Show more replies","G":"Show more likes","Q":"@{0}","W":"{0}…","J":"Save this page before enabling or disabling comments.","e":"{0} Commented {1}.","t":"The comments section will be displayed after the page is published."}')},"J4/A":function(e,t,n){"use strict";n.r(t);var a=n("ut3N"),i=n("2q6Q"),r=n("vlQI"),o=n("y88i"),s=n("Yo/j"),c=n("i8gT"),d=n("2G27"),l=n("xs0R"),u=n("onFi"),f=n("q+wp"),p=function(){function e(e,t){this._loggedLoadSource=!1,this._pageContext=e,this._spHttpClient=t}return e.prototype.createComment=function(e,t){var n=this,a=new i._QosMonitor("CreateCommentAPI");return new Promise(function(i,r){n._createCommentRequest(e,t).then(function(e){n._checkForErrors("CreateCommentAPI",a,e),e.json().then(function(e){i(c.e.toCommentItem(e)),a.writeSuccess()})}).catch(function(e){a.writeUnexpectedFailure("CreateCommentAPI",e),r(e)})})},e.prototype.deleteComment=function(e){var t=this,n=new i._QosMonitor("DeleteCommentAPI");return new Promise(function(a,i){t._deleteCommentRequest(e).then(function(i){t._checkForErrors("DeleteCommentAPI",n,i),a(e),n.writeSuccess()}).catch(function(e){n.writeUnexpectedFailure("DeleteCommentAPI",e),i(e)})})},e.prototype.getComments=function(){return 0!==i._SPPerfExperiment.getVariantAndTrackExperiment(i._PerformanceExperiment.WEXPrefetchCommentsDisabled)?this._getCommentsDataFromAPI():window._spCommentsData?(this._deferClearPageData(),this._logLoadSource("Prefetch"),Promise.resolve(this._convertDataToItemChunk(window._spCommentsData,this._getPageDisabledState()))):Promise.race([this._getCommentsDataFromAPI(),this._getCommentsFromPage()])},e.prototype.getDisabled=function(e){var t=this,n=new i._QosMonitor("GetCommentsDisabledScopeAPI");return new Promise(function(a,i){t._getCommentsDisabledScopeRequest(e).then(function(e){t._checkForErrors("GetCommentsDisabledScopeAPI",n,e),e.json().then(function(e){a(e.value),n.writeSuccess()})}).catch(function(e){n.writeUnexpectedFailure("GetCommentsDisabledScopeAPI",e),i(e)})})},e.prototype.getNext=function(e,t){var n=this;void 0===t&&(t=!1);var a=new i._QosMonitor("LoadMoreCommentsAPI");return new Promise(function(i,r){n._nextCommentsRequest(e,t).then(function(e){n._checkForErrors("LoadMoreCommentsAPI",a,e),e.json().then(function(e){i(n._convertDataToItemChunk(e)),a.writeSuccess()})}).catch(function(e){a.writeUnexpectedFailure("LoadMoreCommentsAPI",e),r(e)})})},e.prototype.getNextLikes=function(e){var t=this,n=new i._QosMonitor("GetCommentLikesAPI");return this._nextLikesRequest(e).then(function(e){return t._checkForErrors("GetPageUrlAPI",n,e),e.json()}).then(function(e){return n.writeSuccess(),Promise.resolve(t._convertServerLikesResponse(e))}).catch(function(e){return n.writeUnexpectedFailure("GetCommentLikesAPI",e),Promise.reject(e)})},e.prototype.likeComment=function(e){var t=this,n=new i._QosMonitor("LikeCommentAPI");return this._likeCommentRequest(e).then(function(e){t._checkLikeResponse("LikeCommentAPI",n,e),n.writeSuccess()}).catch(function(e){return n.writeUnexpectedFailure("LikeCommentAPI",e),Promise.reject(e)})},e.prototype.setDisabled=function(e){var t=this,n=new i._QosMonitor("DisableCommentsAPI");return new Promise(function(a,i){t._disabledRequest(e).then(function(e){t._checkForErrors("DisableCommentsAPI",n,e),a(),n.writeSuccess()}).catch(function(e){n.writeUnexpectedFailure("DisableCommentsAPI",e),i(e)})})},e.prototype.unlikeComment=function(e){var t=this,n=new i._QosMonitor("UnlikeCommentAPI");return this._unlikeCommentRequest(e).then(function(e){t._checkForErrors("UnlikeCommentAPI",n,e),n.writeSuccess()}).catch(function(e){return n.writeUnexpectedFailure("UnlikeCommentAPI",e),Promise.reject(e)})},e.prototype._checkForErrors=function(e,t,n){n||this._throwEmptyResponseError(e,t),f.e.isResponseSuccessful(n.status)||this._throwHttpError(t,n)},e.prototype._checkLikeResponse=function(e,t,n){return n?f.e.isResponseSuccessful(n.status)?void 0:this._throwLikeError(t,n):this._throwEmptyResponseError(e,t)},e.prototype._createCommentRequest=function(e,t){return this._spHttpClient.post(o.Uri.concatenate(this._getCommentsUrl(t),t?"/replies":""),r.SPHttpClient.configurations.v1,{body:JSON.stringify({text:e})})},e.prototype._convertDataToItemChunk=function(e,t){return void 0===t&&(t=0),{comments:e.value.map(function(e){return c.e.toCommentItem(e)}),disabledState:t,count:e["@odata.count"],nextLink:e["@odata.nextLink"]}},e.prototype._deferClearPageData=function(){setTimeout(function(){window._spCommentsData=void 0},0)},e.prototype._convertServerLikesResponse=function(e){return e.value?{likes:e.value,"@odata.nextLink":e["@odata.nextLink"]}:void 0},e.prototype._deleteCommentRequest=function(e){return this._spHttpClient.fetch(o.Uri.concatenate(this._getCommentsUrl(e.id)),r.SPHttpClient.configurations.v1,{method:"delete"})},e.prototype._disabledRequest=function(e){return this._spHttpClient.post(o.Uri.concatenate(this._getPageUrl(),"/SetCommentsDisabled"),r.SPHttpClient.configurations.v1,{body:JSON.stringify({value:e})})},e.prototype._failedResponseFromStatus=function(e){var t,n=((t={})[403]="FailedResponseForbidden",t[404]="FailedResponseNotFound",t[406]="FailedResponseNotAcceptable",t);return n[e]?n[e]:"FailedResponseStatus"},e.prototype._getCommentsDataFromAPI=function(){var t=this;if(0!==i._SPPerfExperiment.getVariantAndTrackExperiment(i._PerformanceExperiment.WEXPrefetchCommentsDisabled))return this._getCommentsFromAPI(void 0).then(function(e){return t._logLoadSource("API"),t._convertDataToItemChunk(e)}).catch(function(t){return a._TraceLogger.logError(e._logSource,t),t});var n=this._spHttpClient.beginBatch(),r=[this._getCommentsFromAPI(n),this.getDisabled(n)];return this._resolveBatch(n,r).then(function(e){return t._logLoadSource("API"),t._convertDataToItemChunk(e[0],e[1])}).catch(function(t){return a._TraceLogger.logError(e._logSource,t),t})},e.prototype._getCommentsFromAPI=function(e){var t=this,n=new i._QosMonitor("LoadCommentsFromAPI");return new Promise(function(a,i){t._getCommentsRequest(e).then(function(e){t._checkForErrors("LoadCommentsFromAPI",n,e),e.json().then(function(e){a(e),n.writeSuccess()})}).catch(function(e){n.writeUnexpectedFailure("LoadCommentsFromAPI",e),i(e)})})},e.prototype._getCommentsFromPage=function(){var e=this;return new Promise(function(t,n){window._spCommentsDataLoaded=function(){t(e._convertDataToItemChunk(window._spCommentsData,e._getPageDisabledState())),e._deferClearPageData(),e._logLoadSource("PrefetchCallback")}})},e.prototype._fields=function(){return l.t.fields},e.prototype._getCommentsDisabledScopeRequest=function(e){return e?e.get(o.Uri.concatenate(this._getPageUrl(),"/CommentsDisabledScope"),r.SPHttpClientBatch.configurations.v1):this._spHttpClient.get(o.Uri.concatenate(this._getPageUrl(),"/CommentsDisabledScope"),r.SPHttpClient.configurations.v1)},e.prototype._getCommentsRequest=function(e){return e?e.get("".concat(this._getCommentsUrl(),"?$expand=replies,likedBy,replies/likedBy&$top=10&$inlineCount=AllPages"),r.SPHttpClientBatch.configurations.v1):this._spHttpClient.get("".concat(this._getCommentsUrl(),"?$expand=replies,likedBy,replies/likedBy&$top=10&$inlineCount=AllPages"),r.SPHttpClient.configurations.v1)},e.prototype._getCommentsUrl=function(e){var t=e?"/Comments".concat(e?"(".concat(e,")"):""):"/GetComments";return o.Uri.concatenate(this._getPageUrl(),t)},e.prototype._getPageDisabledState=function(){if(window._spCommentsDisabledScope)return window._spCommentsDisabledScope.value;var e=3;return window._spCommentsData.value&&window._spCommentsData.value.length&&(e=0),s.e.getDisabledScope(),e},e.prototype._getPageUrl=function(){return o.Uri.concatenate(this._pageContext.legacyPageContext.webServerRelativeUrl||"","/_api/web/lists('".concat(f.e.listId(),"')"),"/GetItemById(".concat(this._itemId(),")"))},e.prototype._itemId=function(){return this._fields().Id||f.e.spPageContextInfo.pageItemId},e.prototype._likeCommentRequest=function(e){return this._toggleLikeCommentRequest(e,!0)},e.prototype._logLoadSource=function(e){this._loggedLoadSource||(this._loggedLoadSource=!0,i._EngagementLogger.logEvent("CommentsLoad.".concat(e)))},e.prototype._nextCommentsRequest=function(e,t){return this._spHttpClient.get("".concat(e)+(t?"&$expand=likedBy":""),r.SPHttpClient.configurations.v1)},e.prototype._nextLikesRequest=function(e){return this._spHttpClient.get(e,r.SPHttpClient.configurations.v1)},e.prototype._resolveBatch=function(e,t){return e.execute().then(function(){return Promise.all(t)})},e.prototype._throwEmptyResponseError=function(e,t){var n=new Error("Response is null, undefined or empty");throw t.writeUnexpectedFailure("FailedResponseStatus",n),n},e.prototype._throwHttpError=function(e,t){var n=new u.e(f.e.getStatusMessage(t,!0),t.status);throw e.writeUnexpectedFailure(this._failedResponseFromStatus(t.status),n),n},e.prototype._throwLikeError=function(e,t){507===t.status?this._throwLikeLimitExceededError(e):this._throwHttpError(e,t)},e.prototype._throwLikeLimitExceededError=function(t){throw t.writeExpectedFailure("LikeCommentAPI",e.likeLimitExceededError),e.likeLimitExceededError},e.prototype._toggleLikeCommentRequest=function(e,t){return this._spHttpClient.fetch(o.Uri.concatenate(this._getCommentsUrl(e),t?"/like":"/unlike"),r.SPHttpClient.configurations.v1,{method:"post",body:JSON.stringify({})})},e.prototype._unlikeCommentRequest=function(e){return this._toggleLikeCommentRequest(e,!1)},e.likeLimitExceededError=new d.e,e._logSource=a._LogSource.create("CommentsDataProvider"),e}();t.default=p},Rrln:function(e,t,n){"use strict";var a=function(){function e(){}return Object.defineProperty(e,"instance",{get:function(){return this._instance||this._createInstance()},enumerable:!1,configurable:!0}),e.isCurrentUser=function(t){var n=e.instance.user;return n&&t&&n.id===t.id},e._createInstance=function(){return this._instance=new e},e.DEFAULT_LIKE_COUNT=0,e}();t.e=a},"Yo/j":function(e,t,n){"use strict";var a=n("ut3N"),i=n("Cgdb"),r=n("jyrw"),o=n("D/Gr"),s=n("J4/A"),c=n("7plR"),d=n("n1SJ"),l=n("F5rs"),u=n("q+wp"),f=n("XdGY"),p=n("xs0R"),m=n("kJuR"),_=function(){function e(){}return e.clearComments=function(){e._dispatch("clear")},e.createComment=function(t,n,a,i){var r=this._createOptimisticComment(t,n,a);c.e.instance.commentsDataProvider().then(function(e){return e.createComment(n,a)}).then(function(t){t.optimisticId=r.optimisticId,e._dispatch("replace",{newComment:t,oldComment:r})}).catch(function(t){null==i||i(),e._dispatch("delete",r),d.e.showError(l.k,void 0,t),e._logError(t)})},e.deleteComment=function(t){e._dispatch("markDelete",t);var n=window.setTimeout(function(){e._dispatch("delete",t)},200);c.e.instance.commentsDataProvider().then(function(e){return e.deleteComment(t)}).catch(function(a){clearTimeout(n),e._dispatch("unmarkDelete",t),d.e.showError(l.x,void 0,a),e._logError(a)})},e.getComments=function(){return c.e.instance.commentsDataProvider().then(function(e){return e.getComments()}).then(function(t){e._dispatch("loaded",t)}).catch(function(t){if(e._logError(t),!Object(f.Xt)())throw t})},e.getDisabledScope=function(){c.e.instance.commentsDataProvider().then(function(e){return e.getDisabled()}).then(function(t){e._dispatch("getDisabled",t)}).catch(e._logError)},e.getNextComments=function(t){return c.e.instance.commentsDataProvider().then(function(e){return e.getNext(t)}).then(function(t){e._dispatch("moreComments",t),u.e.alertScreenreader(l.o,l.s,t.comments.length)}).catch(e._logError)},e.getNextLikes=function(t){c.e.instance.commentsDataProvider().then(function(e){return e.getNextLikes(t.nextLikesLink)}).then(function(n){e._dispatch("moreLikes",{comment:t,likeChunk:n}),u.e.alertScreenreader(l.i,l.r,n.likes.length)}).catch(e._logError)},e.getNextReplies=function(t,n){return c.e.instance.commentsDataProvider().then(function(e){return e.getNext(t,!0)}).then(function(t){t.parentId=n,e._dispatch("moreReplies",t),u.e.alertScreenreader(l.c,l.d,t.comments.length)}).catch(e._logError)},e.focusReply=function(t){e._dispatch("focus",t)},e.likeComment=function(e,t){var n=this;this._dispatch("likeComment",{comment:e,user:t}),c.e.instance.commentsDataProvider().then(function(t){return t.likeComment(e.id)}).then(function(){c.e.instance.a11yManager.alert(l.a)}).catch(function(a){n._dispatch("unlikeComment",{comment:e,user:t}),n._processLikeError(a)})},e.reset=function(){e._dispatch("reset")},e.setDisabled=function(e){return p.t.isCoAuth?(Object(m.r)("IsCommentSectionEnabled",!e),this.setDisabledLocal(e),Promise.resolve()):(this.setDisabledLocal(e),this.setDisabledRemote(e))},e.setDisabledLocal=function(t){e._dispatch("setDisabled",t?1:0)},e.setDisabledRemote=function(t){var n=this;return c.e.instance.commentsDataProvider().then(function(e){return e.setDisabled(t)}).then(function(){return n._dispatch("resetDirty")}).catch(function(n){e._dispatch("setDisabled",t?0:1),e._logError(n)})},e.unlikeComment=function(t,n){this._dispatch("unlikeComment",{comment:t,user:n}),c.e.instance.commentsDataProvider().then(function(e){return e.unlikeComment(t.id)}).then(function(){c.e.instance.a11yManager.alert(l.p)}).catch(function(a){e._dispatch("likeComment",{comment:t,user:n}),d.e.showError(l.j,void 0,a),e._logError(a)})},e._createOptimisticComment=function(t,n,a){var i=new o.e([],new Date,"optimistic_".concat(Math.random()),void 0,a||"0",n,t,0,!1);return e._dispatch("create",i),i},e._dispatch=function(t,n){r.e.dispatch(new i.e(e._actionTypes[t],e._logEntryFor(t),n))},e._logEntryFor=function(t){return new a._LogEntry(e._moduleName,e._logFeatures[t],a._LogType.Event)},e._logError=function(t){a._TraceLogger.logError(e._logSource,t)},e._processLikeError=function(t){var n=t.name===s.default.likeLimitExceededError.name?l.E:l.w;d.e.showError(n,void 0,t),e._logError(t)},e._logSource=a._LogSource.create("CommentActionCreator"),e._moduleName="[CommentActionCreator]",e._logFeatures={clear:"SPPageCommentsCleared",create:"SPPageCommentCreated",delete:"SPPageCommentDeleted",focus:"SPPageCommentReplyFocused",getDisabled:"SPPageCommentsGetDisabled",likeComment:"SPPageCommentLiked",loaded:"SPPageCommentsLoaded",markDelete:"SPPageCommentMarkDeleted",moreComments:"SPPageCommentsMoreCommentsLoaded",moreLikes:"SPPageCommentMoreLikesLoaded",moreReplies:"SPPageCommentsMoreRepliesLoaded",replace:"SPPageCommentReplaced",reset:"SPPageCommentsReset",setDisabled:"SPPageCommentsSetDisabled",unlikeComment:"SPPageCommentUnliked",unmarkDelete:"SPPageCommentMarkUndeleted"},e._actionTypes={clear:"COMMENT_CLEAR",create:"COMMENT_CREATED",delete:"COMMENT_DELETED",focus:"COMMENT_FOCUS_REPLY",getDisabled:"COMMENTS_DISABLED",likeComment:"COMMENT_LIKE",loaded:"COMMENT_CHUNK",markDelete:"COMMENT_MARK_DELETED",moreComments:"COMMENT_CHUNK",moreLikes:"COMMENT_LIKE_CHUNK",moreReplies:"COMMENT_REPLY_CHUNK",replace:"COMMENT_REPLACED",reset:"COMMENT_RESET",setDisabled:"COMMENTS_DISABLED",unlikeComment:"COMMENT_UNLIKE",unmarkDelete:"COMMENT_UNMARK_DELETED",resetDirty:"COMMENT_RESET_DIRTY"},e}();t.e=_},i8gT:function(e,t,n){"use strict";var a=n("D/Gr"),i=n("Rrln"),r=function(){function e(){}return e.allLikesFetched=function(e){return e&&-1===e.indexOf("skiptoken")},e.toCommentItem=function(t){return e._toCommentItemWithLikes(t)},e._toCommentItemWithLikes=function(t){var n=t.replies?t.replies.map(function(t){return e._toCommentItemWithLikes(t)}):[],r=t.likedBy&&t.likedBy.filter(function(e){return e&&!i.e.isCurrentUser(e)}),o=t["likedBy@odata.nextLink"];return e.allLikesFetched(o)&&(o=void 0),new a.e(n,new Date(t.createdDate),t.id,t["replies@odata.nextLink"],t.parentId,t.text,t.author,t.likeCount,t.isLikedByUser,r,o,t,t.mentions)},e}();t.e=r}}]);
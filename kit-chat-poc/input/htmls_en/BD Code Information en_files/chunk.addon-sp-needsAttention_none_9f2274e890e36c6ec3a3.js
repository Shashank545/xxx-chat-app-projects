(window.webpackJsonpf92bf067_bc19_489e_a556_7fe95f508720_0_1_0=window.webpackJsonpf92bf067_bc19_489e_a556_7fe95f508720_0_1_0||[]).push([[74],{"38Zw":function(e,t,n){"use strict";n.r(t),n.d(t,"getNeedsAttentionView",function(){return f});var a=n("17wl"),i=n("ftiL"),r=n("DL2h"),o=n("9kMg"),s=n("jom9"),c=n("GQoE"),d=n("oveG"),l=n("dS01"),u=n("9jhT");function f(e,t,n){var f=l.e.serialize({webAbsoluteUrl:t.webAbsoluteUrl,listFullUrl:t.listFullUrl}),p=n(Object(c.e)({list:d.Pe},f)),m=p&&p.list&&p.list.templateType&&Object(o.r)(p.list.templateType),_=m?"LinkTitle":"LinkFilename",h={},b=[];if(e)for(var g=0,v=e;g<v.length;g++){var y=v[g];if(y.contentTypeId&&y.requiredFields&&y.requiredLookups){for(var S={fieldName:"ContentTypeId",operator:"Eq",values:[y.contentTypeId]},D=r.e(S),I=[],x=y.requiredFields.split(","),C=y.requiredLookups.split(","),O=0;O<x.length;O++){var w=x[O],E="True"===C[O];void 0===h[w]&&(h[w]=E);var A={fieldName:w,lookupId:!!E||void 0,operator:"Eq",values:[""]};I.push(A)}var L=r.t(I,"Or");if(L){var k=r.t([D,L],"And");b.push(k)}}}0===b.length&&b.push(r.e({fieldName:"FileLeafRef",operator:"Eq",values:[""]}));var M=Object.keys(h),P=M.map(function(e){return h[e]}),T=Object(a.__spreadArray)(Object(a.__spreadArray)(["DocIcon",_,"Editor","Modified"],M,!0),["FileDirRef"],!1),U=Object(a.__spreadArray)(Object(a.__spreadArray)([void 0,void 0,void 0,void 0],P,!0),[void 0],!1),F=!!p&&!!p.list&&!!p.list.templateType&&p.list.templateType===o.t.webPageLibrary,H=new i.e("<View Scope='Recursive'/>");H.title=F?s.x:m?s.S:s.g,H.isReadOnly=!0,H.id=u.e,H.replaceFields(T,U),H.updateSort({fieldName:"ID",isAscending:!0}),H.updateRowLimit({rowLimit:30,isPerPage:!0});var R=r.t(b,"Or");return H.addFilters([R]),H}}}]);